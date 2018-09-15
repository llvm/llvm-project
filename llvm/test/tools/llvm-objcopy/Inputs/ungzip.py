import gzip
import sys

with gzip.open(sys.argv[1], 'rb') as f:
  writer = getattr(sys.stdout, 'buffer', None)
  if writer is None:
    writer = sys.stdout
    if sys.platform == "win32":
      import os, msvcrt
      msvcrt.setmode(sys.stdout.fileno(),os.O_BINARY)

  writer.write(f.read())
  sys.stdout.flush()
