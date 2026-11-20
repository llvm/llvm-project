# RUN: llvm-mc -triple=riscv32 -show-encoding --mattr=+experimental-zibi %s \
# RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-ASM
# RUN: llvm-mc -triple=riscv64 -show-encoding --mattr=+experimental-zibi %s \
# RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-ASM
# RUN: llvm-mc --verify -triple=riscv32 -show-encoding %s
# RUN: llvm-mc --verify -triple=riscv64 -show-encoding %s
# RUN: llvm-mc -triple=riscv32 -filetype=obj --mattr=+experimental-zibi %s \
# RUN:        | llvm-objdump -dr --mattr=+experimental-zibi --no-print-imm-hex  - \
# RUN:        | FileCheck %s --check-prefix=CHECK-OBJ
beqi a0, 1, 1024 # expected-error {{instruction requires the following: 'Zibi' (Branch with Immediate)}}
# CHECK-OBJ: beqi a0, 1, 0x400
# CHECK-ASM: beqi a0, 1, 1024
# CHECK-ENCODING: [0x63,0x20,0x15,0x40]
beqi a5, -1, -1024 # expected-error {{instruction requires the following: 'Zibi' (Branch with Immediate)}}
# CHECK-OBJ: beqi a5, -1, 0xfffffc04
# CHECK-ASM: beqi a5, -1, -1024
# CHECK-ENCODING: [0xe3,0xa0,0x07,0xc0]
beqi s0, 22, 0xffe # expected-error {{instruction requires the following: 'Zibi' (Branch with Immediate)}}
# CHECK-OBJ: beqi s0, 22, 0x1006
# CHECK-ASM: beqi s0, 22, 4094
# CHECK-ENCODING: [0xe3,0x2f,0x64,0x7f]
beqi s1, 11, -4096 # expected-error {{instruction requires the following: 'Zibi' (Branch with Immediate)}}
# CHECK-OBJ: beqi s1, 11, 0xfffff00c
# CHECK-ASM: beqi s1, 11, -4096
# CHECK-ENCODING: [0x63,0xa0,0xb4,0x80]
bnei a0, 1, 1024 # expected-error {{instruction requires the following: 'Zibi' (Branch with Immediate)}}
# CHECK-OBJ: bnei a0, 1, 0x410
# CHECK-ASM: bnei a0, 1, 1024
# CHECK-ENCODING: [0x63,0x30,0x15,0x40]
bnei a5, -1, -1024 # expected-error {{instruction requires the following: 'Zibi' (Branch with Immediate)}}
# CHECK-OBJ: bnei a5, -1, 0xfffffc14
# CHECK-ASM: bnei a5, -1, -1024
# CHECK-ENCODING: [0xe3,0xb0,0x07,0xc0]
bnei s0, 22, 0xffe # expected-error {{instruction requires the following: 'Zibi' (Branch with Immediate)}}
# CHECK-OBJ: bnei s0, 22, 0x1016
# CHECK-ASM: bnei s0, 22, 4094
# CHECK-ENCODING: [0xe3,0x3f,0x64,0x7f]
bnei s1, 11, -4096 # expected-error {{instruction requires the following: 'Zibi' (Branch with Immediate)}}
# CHECK-OBJ: bnei s1, 11, 0xfffff01c
# CHECK-ASM: bnei s1, 11, -4096
# CHECK-ENCODING: [0x63,0xb0,0xb4,0x80]

bnei s1, 12, undef+1 # expected-error {{instruction requires the following: 'Zibi' (Branch with Immediate)}}
# CHECK-OBJ:      beqi s1, 12, {{.*}}
# CHECK-OBJ-NEXT: j {{.*}}
# CHECK-OBJ-NEXT: R_RISCV_JAL undef+0x1
# CHECK-ASM:      bnei s1, 12, undef+1
