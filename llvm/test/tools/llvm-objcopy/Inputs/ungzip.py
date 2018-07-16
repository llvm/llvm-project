import gzip
import sys

with gzip.open(sys.argv[1], 'rb') as f:
  sys.stdout.write(f.read())
