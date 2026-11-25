# RUN: llvm-mc %s -triple=riscv64 -mattr=+zfa,+q,+zfh -M no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+zfa,+q,+zfh < %s \
# RUN:     | llvm-objdump --mattr=+zfa,+q,+zfh -M no-aliases -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s
#
# RUN: llvm-mc --verify -triple riscv64 -mattr=+q,+zfh -M no-aliases -show-encoding < %s

# CHECK-ASM-AND-OBJ: fmvh.x.q a1, fs1
# CHECK-ASM: encoding: [0xd3,0x85,0x14,0xe6]
fmvh.x.q a1, fs1 # expected-error {{instruction requires the following: 'Zfa' (Additional Floating-Point)}}

# CHECK-ASM-AND-OBJ: fmvp.q.x fs1, a1, a2
# CHECK-ASM: encoding: [0xd3,0x84,0xc5,0xb6]
fmvp.q.x fs1, a1, a2 # expected-error {{instruction requires the following: 'Zfa' (Additional Floating-Point)}}
