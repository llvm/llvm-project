# RUN: llvm-mc %s -triple=riscv64 -mattr=+c -M no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+c < %s \
# RUN:     | llvm-objdump --mattr=+c --no-print-imm-hex -M no-aliases -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc %s -triple=riscv64 -mattr=+zca -M no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+c < %s \
# RUN:     | llvm-objdump --mattr=+c --no-print-imm-hex -M no-aliases -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s
#
#
# RUN: not llvm-mc -triple riscv64 \
# RUN:     -M no-aliases -show-encoding < %s 2>&1 \
# RUN:     | FileCheck -check-prefixes=CHECK-NO-EXT %s
# RUN: not llvm-mc -triple riscv32 -mattr=+c \
# RUN:     -M no-aliases -show-encoding < %s 2>&1 \
# RUN:     | FileCheck -check-prefixes=CHECK-NO-RV64 %s

# TODO: more exhaustive testing of immediate encoding.

# CHECK-ASM-AND-OBJ: c.ldsp s0, 0(sp)
# CHECK-ASM: encoding: [0x02,0x64]
# CHECK-NO-EXT:  error: instruction requires the following: 'C' (Compressed Instructions) or 'Zca' (part of the C extension, excluding compressed floating point loads/stores){{$}}
# CHECK-NO-RV64:  error: instruction requires the following: 'Zclsd' (Compressed Load/Store pair instructions){{$}}
c.ldsp s0, 0(sp)
# CHECK-ASM-AND-OBJ: c.sdsp s2, 504(sp)
# CHECK-ASM: encoding: [0xca,0xff]
# CHECK-NO-EXT:  error: instruction requires the following: 'C' (Compressed Instructions) or 'Zca' (part of the C extension, excluding compressed floating point loads/stores){{$}}
# CHECK-NO-RV64:  error: instruction requires the following: 'Zclsd' (Compressed Load/Store pair instructions){{$}}
c.sdsp s2, 504(sp)
# CHECK-ASM-AND-OBJ: c.ld a4, 0(a3)
# CHECK-ASM: encoding: [0x98,0x62]
# CHECK-NO-EXT:  error: instruction requires the following: 'C' (Compressed Instructions) or 'Zca' (part of the C extension, excluding compressed floating point loads/stores){{$}}
# CHECK-NO-RV64:  error: instruction requires the following: 'Zclsd' (Compressed Load/Store pair instructions){{$}}
c.ld a4, 0(a3)
# CHECK-ASM-AND-OBJ: c.sd a2, 248(a3)
# CHECK-ASM: encoding: [0xf0,0xfe]
# CHECK-NO-EXT:  error: instruction requires the following: 'C' (Compressed Instructions) or 'Zca' (part of the C extension, excluding compressed floating point loads/stores){{$}}
# CHECK-NO-RV64:  error: instruction requires the following: 'Zclsd' (Compressed Load/Store pair instructions){{$}}
c.sd a2, 248(a3)

# CHECK-ASM-AND-OBJ: c.subw a3, a4
# CHECK-ASM: encoding: [0x99,0x9e]
c.subw a3, a4
# CHECK-ASM-AND-OBJ: c.addw a0, a2
# CHECK-ASM: encoding: [0x31,0x9d]
# CHECK-NO-EXT:  error: instruction requires the following: 'C' (Compressed Instructions) or 'Zca' (part of the C extension, excluding compressed floating point loads/stores){{$}}
# CHECK-NO-RV64:  error: instruction requires the following: RV64I Base Instruction Set{{$}}
c.addw a0, a2

# CHECK-ASM-AND-OBJ: c.addiw a3, -32
# CHECK-ASM: encoding: [0x81,0x36]
# CHECK-NO-EXT:  error: instruction requires the following: 'C' (Compressed Instructions) or 'Zca' (part of the C extension, excluding compressed floating point loads/stores){{$}}
# CHECK-NO-RV64:  error: instruction requires the following: RV64I Base Instruction Set{{$}}
c.addiw a3, -32
# CHECK-ASM-AND-OBJ: c.addiw a3, 31
# CHECK-ASM: encoding: [0xfd,0x26]
# CHECK-NO-EXT:  error: instruction requires the following: 'C' (Compressed Instructions) or 'Zca' (part of the C extension, excluding compressed floating point loads/stores){{$}}
# CHECK-NO-RV64:  error: instruction requires the following: RV64I Base Instruction Set{{$}}
c.addiw a3, 31

# CHECK-ASM-AND-OBJ: c.slli s0, 63
# CHECK-ASM: encoding: [0x7e,0x14]
# CHECK-NO-EXT:  error: instruction requires the following: 'C' (Compressed Instructions) or 'Zca' (part of the C extension, excluding compressed floating point loads/stores){{$}}
c.slli s0, 63
# CHECK-ASM-AND-OBJ: c.srli a3, 63
# CHECK-ASM: encoding: [0xfd,0x92]
# CHECK-NO-EXT:  error: instruction requires the following: 'C' (Compressed Instructions) or 'Zca' (part of the C extension, excluding compressed floating point loads/stores){{$}}
c.srli a3, 63
# CHECK-ASM-AND-OBJ: c.srai a2, 63
# CHECK-ASM: encoding: [0x7d,0x96]
# CHECK-NO-EXT:  error: instruction requires the following: 'C' (Compressed Instructions) or 'Zca' (part of the C extension, excluding compressed floating point loads/stores){{$}}
c.srai a2, 63
