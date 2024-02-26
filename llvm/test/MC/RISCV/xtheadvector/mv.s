# RUN: llvm-mc -triple=riscv64 -show-encoding --mattr=+xtheadvector %s \
# RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: not llvm-mc -triple=riscv64 -show-encoding %s 2>&1 \
# RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+xtheadvector %s \
# RUN:        | llvm-objdump -d --mattr=+xtheadvector --no-print-imm-hex - \
# RUN:        | FileCheck %s --check-prefix=CHECK-INST
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+xtheadvector %s \
# RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

th.vmv.v.v v8, v20
# CHECK-INST: th.vmv.v.v v8, v20
# CHECK-ENCODING: [0x57,0x04,0x0a,0x5e]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 04 0a 5e <unknown>

th.vmv.v.x v8, a0
# CHECK-INST: th.vmv.v.x v8, a0
# CHECK-ENCODING: [0x57,0x44,0x05,0x5e]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 44 05 5e <unknown>

th.vmv.v.i v8, 15
# CHECK-INST: th.vmv.v.i v8, 15
# CHECK-ENCODING: [0x57,0xb4,0x07,0x5e]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 b4 07 5e <unknown>

th.vext.x.v a2, v4, a0
# CHECK-INST: th.vext.x.v a2, v4, a0
# CHECK-ENCODING: [0x57,0x26,0x45,0x32]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 26 45 32 <unknown>

th.vmv.s.x v8, a0
# CHECK-INST: th.vmv.s.x v8, a0
# CHECK-ENCODING: [0x57,0x64,0x05,0x36]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 64 05 36 <unknown>
