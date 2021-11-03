from subprocess import run, PIPE
import re
import sys

llc = run( [ 'llc', '-disable-symbolication','-verify-machineinstrs', '-mtriple=riscv32', '-run-pass=none', '-o', '-', sys.argv[1]], stdout=PIPE, stderr=PIPE )

stdout = llc.stdout.decode()

p = re.compile(r'^\s*%[0-9]+:gpr = ADDI %[0-9]+, 5$', flags=re.MULTILINE)

if (llc.returncode == 0 and p.search(stdout)):
  print('This is interesting!')
  sys.exit(0)
else:
  print('This is NOT interesting!')
  sys.exit(1)
