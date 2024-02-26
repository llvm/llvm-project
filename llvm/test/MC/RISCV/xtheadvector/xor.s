# RUN: llvm-mc -triple=riscv64 -show-encoding --mattr=+xtheadvector %s \
# RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: not llvm-mc -triple=riscv64 -show-encoding %s 2>&1 \
# RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+xtheadvector %s \
# RUN:        | llvm-objdump -d --mattr=+xtheadvector --no-print-imm-hex - \
# RUN:        | FileCheck %s --check-prefix=CHECK-INST
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+xtheadvector %s \
# RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

th.vxor.vv v8, v4, v20, v0.t
# CHECK-INST: th.vxor.vv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x04,0x4a,0x2c]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 04 4a 2c <unknown>

th.vxor.vv v8, v4, v20
# CHECK-INST: th.vxor.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x04,0x4a,0x2e]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 04 4a 2e <unknown>

th.vxor.vx v8, v4, a0, v0.t
# CHECK-INST: th.vxor.vx v8, v4, a0, v0.t
# CHECK-ENCODING: [0x57,0x44,0x45,0x2c]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 44 45 2c <unknown>

th.vxor.vx v8, v4, a0
# CHECK-INST: th.vxor.vx v8, v4, a0
# CHECK-ENCODING: [0x57,0x44,0x45,0x2e]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 44 45 2e <unknown>

th.vxor.vi v8, v4, 15, v0.t
# CHECK-INST: th.vxor.vi v8, v4, 15, v0.t
# CHECK-ENCODING: [0x57,0xb4,0x47,0x2c]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 b4 47 2c <unknown>

th.vxor.vi v8, v4, 15
# CHECK-INST: th.vxor.vi v8, v4, 15
# CHECK-ENCODING: [0x57,0xb4,0x47,0x2e]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 b4 47 2e <unknown>
