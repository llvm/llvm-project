# RUN: llvm-mc %s -triple=riscv64 -mattr=+c,+d -riscv-no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+c,+d < %s \
# RUN:     | llvm-objdump --mattr=+c,+d -M no-aliases -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s
#
# RUN: not llvm-mc -triple riscv64 -mattr=+c \
# RUN:     -riscv-no-aliases -show-encoding < %s 2>&1 \
# RUN:     | FileCheck -check-prefixes=CHECK-NO-EXT-D %s
# RUN: not llvm-mc -triple riscv64 -riscv-no-aliases -show-encoding < %s 2>&1 \
# RUN:     | FileCheck -check-prefixes=CHECK-NO-EXT-DC %s

# CHECK-ASM-AND-OBJ: c.fldsp  fs0, 504(sp)
# CHECK-ASM: encoding: [0x7e,0x34]
# CHECK-NO-EXT-D:  error: instruction requires the following: 'D' (Double-Precision Floating-Point){{$}}
# CHECK-NO-EXT-DC:  error: instruction requires the following: 'C' (Compressed Instructions), 'D' (Double-Precision Floating-Point){{$}}
c.fldsp  fs0, 504(sp)
# CHECK-ASM-AND-OBJ: c.fsdsp  fa7, 504(sp)
# CHECK-ASM: encoding: [0xc6,0xbf]
# CHECK-NO-EXT-D:  error: instruction requires the following: 'D' (Double-Precision Floating-Point){{$}}
# CHECK-NO-EXT-DC:  error: instruction requires the following: 'C' (Compressed Instructions), 'D' (Double-Precision Floating-Point){{$}}
c.fsdsp  fa7, 504(sp)

# CHECK-ASM-AND-OBJ: c.fld  fa3, 248(a5)
# CHECK-ASM: encoding: [0xf4,0x3f]
# CHECK-NO-EXT-D:  error: instruction requires the following: 'D' (Double-Precision Floating-Point){{$}}
# CHECK-NO-EXT-DC:  error: instruction requires the following: 'C' (Compressed Instructions), 'D' (Double-Precision Floating-Point){{$}}
c.fld  fa3, 248(a5)
# CHECK-ASM-AND-OBJ: c.fsd  fa2, 248(a1)
# CHECK-ASM: encoding: [0xf0,0xbd]
# CHECK-NO-EXT-D:  error: instruction requires the following: 'D' (Double-Precision Floating-Point){{$}}
# CHECK-NO-EXT-DC:  error: instruction requires the following: 'C' (Compressed Instructions), 'D' (Double-Precision Floating-Point){{$}}
c.fsd  fa2, 248(a1)
