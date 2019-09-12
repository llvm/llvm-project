import sys

InterestingArgumentPresent = False
FunctionCallPresent = False

input = open(sys.argv[1], "r")
for line in input:
  if "%interesting" in line:
    InterestingArgumentPresent = True
  if "call void @interesting" in line:
    FunctionCallPresent = True

if InterestingArgumentPresent and FunctionCallPresent:
  sys.exit(0) # Interesting!

sys.exit(1)
