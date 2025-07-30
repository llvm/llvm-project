# RUN: llvm-mc %s -triple=riscv64 -mattr=+zfa,+q,+zfh -M no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+zfa,+q,+zfh < %s \
# RUN:     | llvm-objdump --mattr=+zfa,+q,+zfh -M no-aliases -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s
#
# RUN: not llvm-mc -triple riscv64 -mattr=+q,+zfh \
# RUN:     -M no-aliases -show-encoding < %s 2>&1 \
# RUN:     | FileCheck -check-prefixes=CHECK-NO-EXT %s

# CHECK-ASM-AND-OBJ: fmvh.x.q a1, fs1
# CHECK-ASM: encoding: [0xd3,0x85,0x14,0xe6]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fmvh.x.q a1, fs1

# CHECK-ASM-AND-OBJ: fmvp.q.x fs1, a1, a2
# CHECK-ASM: encoding: [0xd3,0x84,0xc5,0xb6]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point){{$}}
fmvp.q.x fs1, a1, a2
