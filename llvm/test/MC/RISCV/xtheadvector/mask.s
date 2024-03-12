# RUN: llvm-mc -triple=riscv64 -show-encoding --mattr=+xtheadvector %s \
# RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: not llvm-mc -triple=riscv64 -show-encoding %s 2>&1 \
# RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+xtheadvector %s \
# RUN:        | llvm-objdump -d --mattr=+xtheadvector - \
# RUN:        | FileCheck %s --check-prefix=CHECK-INST
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+xtheadvector %s \
# RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

th.vmand.mm v8, v4, v20
# CHECK-INST: th.vmand.mm v8, v4, v20
# CHECK-ENCODING: [0x57,0x24,0x4a,0x66]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 24 4a 66 <unknown>

th.vmnand.mm v8, v4, v20
# CHECK-INST: th.vmnand.mm v8, v4, v20
# CHECK-ENCODING: [0x57,0x24,0x4a,0x76]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 24 4a 76 <unknown>

th.vmandnot.mm v8, v4, v20
# CHECK-INST: th.vmandnot.mm v8, v4, v20
# CHECK-ENCODING: [0x57,0x24,0x4a,0x62]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 24 4a 62 <unknown>

th.vmxor.mm v8, v4, v20
# CHECK-INST: th.vmxor.mm v8, v4, v20
# CHECK-ENCODING: [0x57,0x24,0x4a,0x6e]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 24 4a 6e <unknown>

th.vmor.mm v8, v4, v20
# CHECK-INST: th.vmor.mm v8, v4, v20
# CHECK-ENCODING: [0x57,0x24,0x4a,0x6a]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 24 4a 6a <unknown>

th.vmnor.mm v8, v4, v20
# CHECK-INST: th.vmnor.mm v8, v4, v20
# CHECK-ENCODING: [0x57,0x24,0x4a,0x7a]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 24 4a 7a <unknown>

th.vmornot.mm v8, v4, v20
# CHECK-INST: th.vmornot.mm v8, v4, v20
# CHECK-ENCODING: [0x57,0x24,0x4a,0x72]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 24 4a 72 <unknown>

th.vmxnor.mm v8, v4, v20
# CHECK-INST: th.vmxnor.mm v8, v4, v20
# CHECK-ENCODING: [0x57,0x24,0x4a,0x7e]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 24 4a 7e <unknown>

th.vmpopc.m a2, v4, v0.t
# CHECK-INST: th.vmpopc.m a2, v4, v0.t
# CHECK-ENCODING: [0x57,0x26,0x40,0x50]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 26 40 50 <unknown>

th.vmpopc.m a2, v4
# CHECK-INST: th.vmpopc.m a2, v4
# CHECK-ENCODING: [0x57,0x26,0x40,0x52]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 26 40 52 <unknown>

th.vmfirst.m a2, v4, v0.t
# CHECK-INST: th.vmfirst.m a2, v4, v0.t
# CHECK-ENCODING: [0x57,0x26,0x40,0x54]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 26 40 54 <unknown>

th.vmfirst.m a2, v4
# CHECK-INST: th.vmfirst.m a2, v4
# CHECK-ENCODING: [0x57,0x26,0x40,0x56]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 26 40 56 <unknown>

th.vmsbf.m v8, v4, v0.t
# CHECK-INST: th.vmsbf.m v8, v4, v0.t
# CHECK-ENCODING: [0x57,0xa4,0x40,0x58]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 a4 40 58 <unknown>

th.vmsbf.m v8, v4
# CHECK-INST: th.vmsbf.m v8, v4
# CHECK-ENCODING: [0x57,0xa4,0x40,0x5a]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 a4 40 5a <unknown>

th.vmsif.m v8, v4, v0.t
# CHECK-INST: th.vmsif.m v8, v4, v0.t
# CHECK-ENCODING: [0x57,0xa4,0x41,0x58]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 a4 41 58 <unknown>

th.vmsif.m v8, v4
# CHECK-INST: th.vmsif.m v8, v4
# CHECK-ENCODING: [0x57,0xa4,0x41,0x5a]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 a4 41 5a <unknown>

th.vmsof.m v8, v4, v0.t
# CHECK-INST: th.vmsof.m v8, v4, v0.t
# CHECK-ENCODING: [0x57,0x24,0x41,0x58]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 24 41 58 <unknown>

th.vmsof.m v8, v4
# CHECK-INST: th.vmsof.m v8, v4
# CHECK-ENCODING: [0x57,0x24,0x41,0x5a]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 24 41 5a <unknown>

th.viota.m v8, v4, v0.t
# CHECK-INST: th.viota.m v8, v4, v0.t
# CHECK-ENCODING: [0x57,0x24,0x48,0x58]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 24 48 58 <unknown>

th.viota.m v8, v4
# CHECK-INST: th.viota.m v8, v4
# CHECK-ENCODING: [0x57,0x24,0x48,0x5a]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 24 48 5a <unknown>

th.vid.v v8, v0.t
# CHECK-INST: th.vid.v v8, v0.t
# CHECK-ENCODING: [0x57,0xa4,0x08,0x58]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 a4 08 58 <unknown>

th.vid.v v8
# CHECK-INST: th.vid.v v8
# CHECK-ENCODING: [0x57,0xa4,0x08,0x5a]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 a4 08 5a <unknown>
