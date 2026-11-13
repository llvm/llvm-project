# RUN: llvm-mc %s -triple=riscv64 -mattr=+c,+d -M no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+c,+d < %s \
# RUN:     | llvm-objdump --no-print-imm-hex --mattr=+c,+d -M no-aliases -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc %s -triple=riscv64 -mattr=+zcd,+d -M no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+zcd,+d < %s \
# RUN:     | llvm-objdump --no-print-imm-hex --mattr=+zcd,+d -M no-aliases -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s
#
# RUN: llvm-mc --verify -triple riscv64 -mattr=+c -M no-aliases -show-encoding < %s
# RUN: llvm-mc --verify -triple riscv64 -M no-aliases -show-encoding < %s

# CHECK-ASM-AND-OBJ: c.fldsp  fs0, 504(sp)
# CHECK-ASM: encoding: [0x7e,0x34]
c.fldsp  fs0, 504(sp) # expected-error {{instruction requires the following: 'C' (Compressed Instructions) and 'D' (Double-Precision Floating-Point) or 'Zcd' (Compressed Double-Precision Floating-Point Instructions)}}
# CHECK-ASM-AND-OBJ: c.fsdsp  fa7, 504(sp)
# CHECK-ASM: encoding: [0xc6,0xbf]
c.fsdsp  fa7, 504(sp) # expected-error {{instruction requires the following: 'C' (Compressed Instructions) and 'D' (Double-Precision Floating-Point) or 'Zcd' (Compressed Double-Precision Floating-Point Instructions)}}

# CHECK-ASM-AND-OBJ: c.fld  fa3, 248(a5)
# CHECK-ASM: encoding: [0xf4,0x3f]
c.fld  fa3, 248(a5) # expected-error {{instruction requires the following: 'C' (Compressed Instructions) and 'D' (Double-Precision Floating-Point) or 'Zcd' (Compressed Double-Precision Floating-Point Instructions)}}
# CHECK-ASM-AND-OBJ: c.fsd  fa2, 248(a1)
# CHECK-ASM: encoding: [0xf0,0xbd]
c.fsd  fa2, 248(a1) # expected-error {{instruction requires the following: 'C' (Compressed Instructions) and 'D' (Double-Precision Floating-Point) or 'Zcd' (Compressed Double-Precision Floating-Point Instructions)}}
