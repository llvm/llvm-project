import sys

FunctionCallPresent = False

input = open(sys.argv[1], "r")
for line in input:
  if "call void @use" in line:
    FunctionCallPresent = True

if FunctionCallPresent:
  sys.exit(0) # Interesting!

sys.exit(1)
