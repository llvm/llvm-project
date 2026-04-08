# RUN: llvm-mc -triple=riscv32 -show-encoding --mattr=+v --mattr=+experimental-zvabd %s \
# RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: not llvm-mc -triple=riscv32 -show-encoding %s 2>&1 \
# RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
# RUN: llvm-mc -triple=riscv32 -filetype=obj --mattr=+v --mattr=+experimental-zvabd %s \
# RUN:        | llvm-objdump -d --mattr=+v --mattr=+experimental-zvabd --no-print-imm-hex  - \
# RUN:        | FileCheck %s --check-prefix=CHECK-INST
# RUN: llvm-mc -triple=riscv32 -filetype=obj --mattr=+v --mattr=+experimental-zvabd %s \
# RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

vabs.v v9, v8
# CHECK-INST: vabs.v v9, v8
# CHECK-ENCODING: [0xd7,0x24,0x88,0x4a]
# CHECK-ERROR: instruction requires the following: 'Zvabd' (Vector Absolute Difference){{$}}
# CHECK-UNKNOWN: 4a8824d7 <unknown>

vabd.vv v10, v9, v8
# CHECK-INST: vabd.vv v10, v9, v8
# CHECK-ENCODING: [0x57,0x25,0x94,0x46]
# CHECK-ERROR: instruction requires the following: 'Zvabd' (Vector Absolute Difference){{$}}
# CHECK-UNKNOWN: 46942557 <unknown>

vabd.vv v10, v9, v8, v0.t
# CHECK-INST: vabd.vv v10, v9, v8, v0.t
# CHECK-ENCODING: [0x57,0x25,0x94,0x44]
# CHECK-ERROR: instruction requires the following: 'Zvabd' (Vector Absolute Difference){{$}}
# CHECK-UNKNOWN: 44942557 <unknown>

vabdu.vv v10, v9, v8
# CHECK-INST: vabdu.vv v10, v9, v8
# CHECK-ENCODING: [0x57,0x25,0x94,0x4e]
# CHECK-ERROR: instruction requires the following: 'Zvabd' (Vector Absolute Difference){{$}}
# CHECK-UNKNOWN: 4e942557 <unknown>

vabdu.vv v10, v9, v8, v0.t
# CHECK-INST: vabdu.vv v10, v9, v8, v0.t
# CHECK-ENCODING: [0x57,0x25,0x94,0x4c]
# CHECK-ERROR: instruction requires the following: 'Zvabd' (Vector Absolute Difference){{$}}
# CHECK-UNKNOWN: 4c942557 <unknown>

vwabda.vv v10, v9, v8
# CHECK-INST: vwabda.vv v10, v9, v8
# CHECK-ENCODING: [0x57,0x25,0x94,0x56]
# CHECK-ERROR: instruction requires the following: 'Zvabd' (Vector Absolute Difference){{$}}
# CHECK-UNKNOWN: 56942557 <unknown>

vwabda.vv v10, v9, v8, v0.t
# CHECK-INST: vwabda.vv v10, v9, v8, v0.t
# CHECK-ENCODING: [0x57,0x25,0x94,0x54]
# CHECK-ERROR: instruction requires the following: 'Zvabd' (Vector Absolute Difference){{$}}
# CHECK-UNKNOWN: 54942557 <unknown>

vwabdau.vv v10, v9, v8
# CHECK-INST: vwabdau.vv v10, v9, v8
# CHECK-ENCODING: [0x57,0x25,0x94,0x5a]
# CHECK-ERROR: instruction requires the following: 'Zvabd' (Vector Absolute Difference){{$}}
# CHECK-UNKNOWN: 5a942557 <unknown>

vwabdau.vv v10, v9, v8, v0.t
# CHECK-INST: vwabdau.vv v10, v9, v8, v0.t
# CHECK-ENCODING: [0x57,0x25,0x94,0x58]
# CHECK-ERROR: instruction requires the following: 'Zvabd' (Vector Absolute Difference){{$}}
# CHECK-UNKNOWN: 58942557 <unknown>
