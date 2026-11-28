# Verify that UPMUCNT0-7 are parsed and disassembled correctly.

# RUN: llvm-mc -triple=hexagon -mv79 -filetype=obj %s | llvm-objdump -d - | FileCheck %s

r0=upmucnt0
r1=upmucnt1
r2=upmucnt2
r3=upmucnt3
r4=upmucnt4
r5=upmucnt5
r6=upmucnt6
r7=upmucnt7

# CHECK:      6a14c000 { r0 = upmucnt0 }
# CHECK-NEXT: 6a15c001 { r1 = upmucnt1 }
# CHECK-NEXT: 6a16c002 { r2 = upmucnt2 }
# CHECK-NEXT: 6a17c003 { r3 = upmucnt3 }
# CHECK-NEXT: 6a18c004 { r4 = upmucnt4 }
# CHECK-NEXT: 6a19c005 { r5 = upmucnt5 }
# CHECK-NEXT: 6a1ac006 { r6 = upmucnt6 }
# CHECK-NEXT: 6a1bc007 { r7 = upmucnt7 }
