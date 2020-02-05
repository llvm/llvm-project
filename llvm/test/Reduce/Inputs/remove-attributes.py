import sys

for line in open(sys.argv[1], "r"):
  if "use-soft-float" in line:
    sys.exit(0) # Interesting!

sys.exit(1)
