# RUN: llvm-mc %s -triple=riscv32 -mattr=+zfa,+d,+zfh -riscv-no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+zfa,+d,+zfh < %s \
# RUN:     | llvm-objdump --mattr=+zfa,+d,+zfh -M no-aliases -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s
#
# RUN: not llvm-mc -triple riscv32 -mattr=+d,+zfh \
# RUN:     -riscv-no-aliases -show-encoding < %s 2>&1 \
# RUN:     | FileCheck -check-prefixes=CHECK-NO-EXT %s

# CHECK-ASM-AND-OBJ: fmvh.x.d a1, fs1
# CHECK-ASM: encoding: [0xd3,0x85,0x14,0xe2]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fmvh.x.d a1, fs1

# CHECK-ASM-AND-OBJ: fmvp.d.x fs1, a1, a2
# CHECK-ASM: encoding: [0xd3,0x84,0xc5,0xb2]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fmvp.d.x fs1, a1, a2
