##################################################
# Utility Function to Split Image into Four Quadrants
##################################################
def split_image(image):
    """
    Divides an image into four equal quadrants: top-left, top-right, bottom-left, and bottom-right.
    """
    width, height = image.size
    left_half = width // 2
    top_half = height // 2

    top_left = image.crop((0, 0, left_half, top_half))
    top_right = image.crop((left_half, 0, width, top_half))
    bottom_left = image.crop((0, top_half, left_half, height))
    bottom_right = image.crop((left_half, top_half, width, height))

    return [top_left, top_right, bottom_left, bottom_right]
