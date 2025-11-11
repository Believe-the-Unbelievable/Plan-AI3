import json
import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ---------------- Load Dataset ----------------
with open("plan_db_simple.json", "r") as f:
    plans = json.load(f)

X = np.array([p["uiiv"] for p in plans], dtype=float)

# ---------------- User Input ----------------
print("=" * 50)
print("🏠 HOUSE PLAN RECOMMENDATION SYSTEM")
print("=" * 50)

# Get plot dimensions
while True:
    try:
        width = float(input("\n📏 Enter plot width (in meters): "))
        height = float(input("📏 Enter plot height (in meters): "))
        if width > 0 and height > 0:
            break
        else:
            print("❌ Width and height must be positive numbers!")
    except ValueError:
        print("❌ Please enter valid numbers!")

# Get room requirements
print("\n🚪 Enter number of rooms needed:")
print("-" * 50)

room_counts = {}
room_types = [
    ("Bedroom", "🛏️"),
    ("Toilet", "🚽"),
    ("Parking", "🚗"),
    ("Sitout", "🪑"),
    ("Living Room", "🛋️"),
    ("Kitchen", "🍳"),
    ("Dining", "🍽️"),   
]

for room_name, emoji in room_types:
    while True:
        try:
            count = int(input(f"{emoji} {room_name}: "))
            if count >= 0:
                room_counts[room_name.lower().replace(" ", "_")] = count
                break
            else:
                print("❌ Please enter a non-negative number!")
        except ValueError:
            print("❌ Please enter a valid integer!")

# Construct user UIIV vector
# Format: [width, height, parking, sitout, living_room, bedroom, kitchen, dining, toilet]
user_uiiv = np.array([
    width,
    height,
    room_counts["parking"],
    room_counts["sitout"],
    room_counts["living_room"],
    room_counts["bedroom"],
    room_counts["kitchen"],
    room_counts["dining"],
    room_counts["toilet"]
], dtype=float)

print("\n" + "=" * 50)
print("🔍 Searching for the best matching plan...")
print("=" * 50)

# ---------------- KMeans + KNN Plan Selection ----------------
n_clusters = min(4, len(plans))
km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42).fit(X)
target_cluster = km.predict(user_uiiv.reshape(1, -1))[0]
cluster_indices = np.where(km.labels_ == target_cluster)[0]

Xc = X[cluster_indices]
nn = NearestNeighbors(n_neighbors=1, metric="euclidean").fit(Xc)
dists, idxs = nn.kneighbors(user_uiiv.reshape(1, -1))

best_plan = plans[cluster_indices[idxs[0][0]]]

# Display results
print(f"\n✅ Selected Plan: {best_plan['plan_id']}")
print(f"📐 Plot Size: {best_plan['plot_width']}m × {best_plan['plot_height']}m")
print(f"📊 Match Distance: {dists[0][0]:.2f}")
print("\n🏠 Rooms in this plan:")
for room_name in best_plan["rooms"].keys():
    print(f"   • {room_name}")

# ---------------- Visualization ----------------
rooms = best_plan["rooms"]
plot_w = best_plan["plot_width"]
plot_h = best_plan["plot_height"]

fig, ax = plt.subplots(figsize=(10, 8))

# Color map for different room types
color_map = {
    "bedroom": "#FFB6C1",
    "toilet": "#D3D3D3",
    "parking": "#FFE4B5",
    "sitout": "#E0F8E0",
    "living": "#B0E0E6",
    "kitchen": "#F0E68C",
    "dining": "#DDA0DD"
}

# Draw each room
for name, (x, y, w, h) in rooms.items():
    # Determine color based on room type
    room_type = name.lower().split()[0] if " " in name else name.lower()
    fill_color = color_map.get(room_type, "#F5F5F5")
    
    ax.add_patch(patches.Rectangle((x, y), w, h, 
                                   fill=True, 
                                   facecolor=fill_color,
                                   edgecolor="black", 
                                   linewidth=1.8,
                                   alpha=0.7))
    ax.text(x + w / 2, y + h / 2, name, 
           ha="center", va="center", 
           fontsize=9, weight="bold")
    ax.text(x + w / 2, y + 0.2, f"{w}m × {h}m", 
           ha="center", va="bottom", 
           fontsize=7, color="gray")

# Draw total boundary
ax.add_patch(patches.Rectangle((0, 0), plot_w, plot_h, 
                               fill=False, 
                               edgecolor="red", 
                               linewidth=2.5))

# Add labels
ax.text(plot_w / 2, -0.5, f"Total Plot: {plot_w}m × {plot_h}m", 
       ha="center", fontsize=10, weight="bold", color="red")

# Styling
ax.set_xlim(-1, plot_w + 1)
ax.set_ylim(-1.5, plot_h + 1)
ax.set_aspect("equal")
ax.axis("off")
plt.title(f"🏠 Recommended Plan: {best_plan['plan_id']}", 
         fontsize=14, weight="bold", pad=20)
plt.tight_layout()
plt.show()

print("\n" + "=" * 50)
print("✨ Visualization complete!")
print("=" * 50)