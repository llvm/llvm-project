# RUN: llvm-mc -triple=riscv32 -show-encoding --mattr=+experimental-zibi %s \
# RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-ASM
# RUN: llvm-mc -triple=riscv64 -show-encoding --mattr=+experimental-zibi %s \
# RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-ASM
# RUN: not llvm-mc -triple=riscv32 -show-encoding %s 2>&1 \
# RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
# RUN: not llvm-mc -triple=riscv64 -show-encoding %s 2>&1 \
# RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
# RUN: llvm-mc -triple=riscv32 -filetype=obj --mattr=+experimental-zibi %s \
# RUN:        | llvm-objdump -d --mattr=+experimental-zibi --no-print-imm-hex  - \
# RUN:        | FileCheck %s --check-prefix=CHECK-OBJ
# RUN: llvm-mc -triple=riscv32 -filetype=obj --mattr=+experimental-zibi %s \
# RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+experimental-zibi %s \
# RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN
beqi a0, 1, 1024
# CHECK-OBJ: beqi a0, 1, 0x400
# CHECK-ASM: beqi a0, 1, 1024
# CHECK-ENCODING: [0x63,0x20,0x15,0x40]
# CHECK-ERROR: instruction requires the following: 'Zibi' (Branch with Immediate){{$}}
# CHECK-UNKNOWN: 40152063 <unknown>
beqi a5, -1, -1024
# CHECK-OBJ: beqi a5, -1, 0xfffffc04
# CHECK-ASM: beqi a5, -1, -1024
# CHECK-ENCODING: [0xe3,0xa0,0x07,0xc0]
# CHECK-ERROR: instruction requires the following: 'Zibi' (Branch with Immediate){{$}}
# CHECK-UNKNOWN: c007a0e3 <unknown>
beqi s0, 22, 0xffe
# CHECK-OBJ: beqi s0, 22, 0x1006
# CHECK-ASM: beqi s0, 22, 4094
# CHECK-ENCODING: [0xe3,0x2f,0x64,0x7f]
# CHECK-ERROR: instruction requires the following: 'Zibi' (Branch with Immediate){{$}}
# CHECK-UNKNOWN: 7f642fe3 <unknown>
beqi s1, 11, -4096
# CHECK-OBJ: beqi s1, 11, 0xfffff00c
# CHECK-ASM: beqi s1, 11, -4096
# CHECK-ENCODING: [0x63,0xa0,0xb4,0x80]
# CHECK-ERROR: instruction requires the following: 'Zibi' (Branch with Immediate){{$}}
# CHECK-UNKNOWN: 80b4a063 <unknown>
bnei a0, 1, 1024
# CHECK-OBJ: bnei a0, 1, 0x410
# CHECK-ASM: bnei a0, 1, 1024
# CHECK-ENCODING: [0x63,0x30,0x15,0x40]
# CHECK-ERROR: instruction requires the following: 'Zibi' (Branch with Immediate){{$}}
# CHECK-UNKNOWN: 40153063 <unknown>
bnei a5, -1, -1024
# CHECK-OBJ: bnei a5, -1, 0xfffffc14
# CHECK-ASM: bnei a5, -1, -1024
# CHECK-ENCODING: [0xe3,0xb0,0x07,0xc0]
# CHECK-ERROR: instruction requires the following: 'Zibi' (Branch with Immediate){{$}}
# CHECK-UNKNOWN: c007b0e3 <unknown>
bnei s0, 22, 0xffe
# CHECK-OBJ: bnei s0, 22, 0x1016
# CHECK-ASM: bnei s0, 22, 4094
# CHECK-ENCODING: [0xe3,0x3f,0x64,0x7f]
# CHECK-ERROR: instruction requires the following: 'Zibi' (Branch with Immediate){{$}}
# CHECK-UNKNOWN: 7f643fe3 <unknown>
bnei s1, 11, -4096
# CHECK-OBJ: bnei s1, 11, 0xfffff01c
# CHECK-ASM: bnei s1, 11, -4096
# CHECK-ENCODING: [0x63,0xb0,0xb4,0x80]
# CHECK-ERROR: instruction requires the following: 'Zibi' (Branch with Immediate){{$}}
# CHECK-UNKNOWN: 80b4b063 <unknown>
