# Verify that UPMUCNT0-7 cannot be written.

# RUN: not llvm-mc -triple=hexagon -mv79 -filetype=asm %s 2>&1 | FileCheck %s

upmucnt0=r0
upmucnt1=r0
upmucnt2=r0
upmucnt3=r0
upmucnt4=r0
upmucnt5=r0
upmucnt6=r0
upmucnt7=r0

# CHECK: error: Cannot write to read-only register `UPMUCNT0'
# CHECK: error: Cannot write to read-only register `UPMUCNT1'
# CHECK: error: Cannot write to read-only register `UPMUCNT2'
# CHECK: error: Cannot write to read-only register `UPMUCNT3'
# CHECK: error: Cannot write to read-only register `UPMUCNT4'
# CHECK: error: Cannot write to read-only register `UPMUCNT5'
# CHECK: error: Cannot write to read-only register `UPMUCNT6'
# CHECK: error: Cannot write to read-only register `UPMUCNT7'
