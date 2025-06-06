# RUN: llvm-mc %s -triple=riscv32 -mattr=+zihintntl,+c -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc %s -triple=riscv64 -mattr=+zihintntl,+c -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+zihintntl,+c < %s \
# RUN:     | llvm-objdump --mattr=+zihintntl,+c -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+zihintntl,+c < %s \
# RUN:     | llvm-objdump --mattr=+zihintntl,+c -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s
# RUN: not llvm-mc %s -triple=riscv32 -mattr=+zihintntl 2>&1 | FileCheck -check-prefix=CHECK-NO-C %s
# RUN: not llvm-mc %s -triple=riscv64 -mattr=+zihintntl 2>&1 | FileCheck -check-prefix=CHECK-NO-C %s

# CHECK-ASM-AND-OBJ: ntl.p1
# CHECK-ASM: encoding: [0x33,0x00,0x20,0x00]
ntl.p1

# CHECK-ASM-AND-OBJ: ntl.pall
# CHECK-ASM: encoding: [0x33,0x00,0x30,0x00]
ntl.pall

# CHECK-ASM-AND-OBJ: ntl.s1
# CHECK-ASM: encoding: [0x33,0x00,0x40,0x00]
ntl.s1

# CHECK-ASM-AND-OBJ: ntl.all
# CHECK-ASM: encoding: [0x33,0x00,0x50,0x00]
ntl.all

# CHECK-ASM-AND-OBJ: c.ntl.p1
# CHECK-ASM: encoding: [0x0a,0x90]
# CHECK-NO-C: error: instruction requires the following: 'C' (Compressed Instructions)
# CHECK-NO-C-NEXT: c.ntl.p1
c.ntl.p1

# CHECK-ASM-AND-OBJ: c.ntl.pall
# CHECK-ASM: encoding: [0x0e,0x90]
# CHECK-NO-C: error: instruction requires the following: 'C' (Compressed Instructions)
# CHECK-NO-C-NEXT: c.ntl.pall
c.ntl.pall

# CHECK-ASM-AND-OBJ: c.ntl.s1
# CHECK-ASM: encoding: [0x12,0x90]
# CHECK-NO-C: error: instruction requires the following: 'C' (Compressed Instructions)
# CHECK-NO-C-NEXT: c.ntl.s1
c.ntl.s1

# CHECK-ASM-AND-OBJ: c.ntl.all
# CHECK-ASM: encoding: [0x16,0x90]
# CHECK-NO-C: error: instruction requires the following: 'C' (Compressed Instructions)
# CHECK-NO-C-NEXT: c.ntl.all
c.ntl.all
