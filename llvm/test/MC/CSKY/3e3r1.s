# RUN: llvm-mc %s -triple=csky -show-encoding -csky-no-aliases -mattr=+3e3r1 \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=csky -mattr=+3e3r1 < %s \
# RUN:     | llvm-objdump --mattr=+3e3r1 -M no-aliases -M abi-names -d -r - \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM-AND-OBJ %s


# CHECK-ASM-AND-OBJ: mul.s32 a3, l0, a1
# CHECK-ASM: encoding: [0x24,0xf8,0x03,0x82]
mul.s32 a3, l0, a1

# CHECK-ASM-AND-OBJ: mul.u32 a3, l0, a1
# CHECK-ASM: encoding: [0x24,0xf8,0x03,0x80]
mul.u32 a3, l0, a1

# CHECK-ASM-AND-OBJ: mula.s32 a3, l0, a1
# CHECK-ASM: encoding: [0x24,0xf8,0x83,0x82]
mula.s32 a3, l0, a1

# CHECK-ASM-AND-OBJ: mula.u32 a3, l0, a1
# CHECK-ASM: encoding: [0x24,0xf8,0x83,0x80]
mula.u32 a3, l0, a1