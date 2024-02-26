# RUN: llvm-mc -triple=riscv64 -show-encoding --mattr=+xtheadvector %s \
# RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING
# RUN: not llvm-mc -triple=riscv64 -show-encoding %s 2>&1 \
# RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+xtheadvector %s \
# RUN:        | llvm-objdump -d --mattr=+xtheadvector --no-print-imm-hex - \
# RUN:        | FileCheck %s --check-prefix=CHECK-INST
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+xtheadvector %s \
# RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

th.vmslt.vv v0, v4, v20, v0.t
# CHECK-INST: th.vmslt.vv v0, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x00,0x4a,0x6c]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 00 4a 6c <unknown>

th.vmseq.vv v8, v4, v20, v0.t
# CHECK-INST: th.vmseq.vv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x04,0x4a,0x60]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 04 4a 60 <unknown>

th.vmseq.vv v8, v4, v20
# CHECK-INST: th.vmseq.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x04,0x4a,0x62]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 04 4a 62 <unknown>

th.vmseq.vx v8, v4, a0, v0.t
# CHECK-INST: th.vmseq.vx v8, v4, a0, v0.t
# CHECK-ENCODING: [0x57,0x44,0x45,0x60]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 44 45 60 <unknown>

th.vmseq.vx v8, v4, a0
# CHECK-INST: th.vmseq.vx v8, v4, a0
# CHECK-ENCODING: [0x57,0x44,0x45,0x62]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 44 45 62 <unknown>

th.vmseq.vi v8, v4, 15, v0.t
# CHECK-INST: th.vmseq.vi v8, v4, 15, v0.t
# CHECK-ENCODING: [0x57,0xb4,0x47,0x60]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 b4 47 60 <unknown>

th.vmseq.vi v8, v4, 15
# CHECK-INST: th.vmseq.vi v8, v4, 15
# CHECK-ENCODING: [0x57,0xb4,0x47,0x62]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 b4 47 62 <unknown>

th.vmsne.vv v8, v4, v20, v0.t
# CHECK-INST: th.vmsne.vv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x04,0x4a,0x64]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 04 4a 64 <unknown>

th.vmsne.vv v8, v4, v20
# CHECK-INST: th.vmsne.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x04,0x4a,0x66]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 04 4a 66 <unknown>

th.vmsne.vx v8, v4, a0, v0.t
# CHECK-INST: th.vmsne.vx v8, v4, a0, v0.t
# CHECK-ENCODING: [0x57,0x44,0x45,0x64]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 44 45 64 <unknown>

th.vmsne.vx v8, v4, a0
# CHECK-INST: th.vmsne.vx v8, v4, a0
# CHECK-ENCODING: [0x57,0x44,0x45,0x66]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 44 45 66 <unknown>

th.vmsne.vi v8, v4, 15, v0.t
# CHECK-INST: th.vmsne.vi v8, v4, 15, v0.t
# CHECK-ENCODING: [0x57,0xb4,0x47,0x64]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 b4 47 64 <unknown>

th.vmsne.vi v8, v4, 15
# CHECK-INST: th.vmsne.vi v8, v4, 15
# CHECK-ENCODING: [0x57,0xb4,0x47,0x66]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 b4 47 66 <unknown>

th.vmsltu.vv v8, v4, v20, v0.t
# CHECK-INST: th.vmsltu.vv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x04,0x4a,0x68]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 04 4a 68 <unknown>

th.vmsltu.vv v8, v4, v20
# CHECK-INST: th.vmsltu.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x04,0x4a,0x6a]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 04 4a 6a <unknown>

th.vmsltu.vx v8, v4, a0, v0.t
# CHECK-INST: th.vmsltu.vx v8, v4, a0, v0.t
# CHECK-ENCODING: [0x57,0x44,0x45,0x68]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 44 45 68 <unknown>

th.vmsltu.vx v8, v4, a0
# CHECK-INST: th.vmsltu.vx v8, v4, a0
# CHECK-ENCODING: [0x57,0x44,0x45,0x6a]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 44 45 6a <unknown>

th.vmslt.vv v8, v4, v20, v0.t
# CHECK-INST: th.vmslt.vv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x04,0x4a,0x6c]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 04 4a 6c <unknown>

th.vmslt.vv v8, v4, v20
# CHECK-INST: th.vmslt.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x04,0x4a,0x6e]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 04 4a 6e <unknown>

th.vmslt.vx v8, v4, a0, v0.t
# CHECK-INST: th.vmslt.vx v8, v4, a0, v0.t
# CHECK-ENCODING: [0x57,0x44,0x45,0x6c]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 44 45 6c <unknown>

th.vmslt.vx v8, v4, a0
# CHECK-INST: th.vmslt.vx v8, v4, a0
# CHECK-ENCODING: [0x57,0x44,0x45,0x6e]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 44 45 6e <unknown>

th.vmsleu.vv v8, v4, v20, v0.t
# CHECK-INST: th.vmsleu.vv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x04,0x4a,0x70]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 04 4a 70 <unknown>

th.vmsleu.vv v8, v4, v20
# CHECK-INST: th.vmsleu.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x04,0x4a,0x72]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 04 4a 72 <unknown>

th.vmsleu.vx v8, v4, a0, v0.t
# CHECK-INST: th.vmsleu.vx v8, v4, a0, v0.t
# CHECK-ENCODING: [0x57,0x44,0x45,0x70]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 44 45 70 <unknown>

th.vmsleu.vx v8, v4, a0
# CHECK-INST: th.vmsleu.vx v8, v4, a0
# CHECK-ENCODING: [0x57,0x44,0x45,0x72]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 44 45 72 <unknown>

th.vmsleu.vi v8, v4, 15, v0.t
# CHECK-INST: th.vmsleu.vi v8, v4, 15, v0.t
# CHECK-ENCODING: [0x57,0xb4,0x47,0x70]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 b4 47 70 <unknown>

th.vmsleu.vi v8, v4, 15
# CHECK-INST: th.vmsleu.vi v8, v4, 15
# CHECK-ENCODING: [0x57,0xb4,0x47,0x72]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 b4 47 72 <unknown>

th.vmsle.vv v8, v4, v20, v0.t
# CHECK-INST: th.vmsle.vv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x04,0x4a,0x74]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 04 4a 74 <unknown>

th.vmsle.vv v8, v4, v20
# CHECK-INST: th.vmsle.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x04,0x4a,0x76]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 04 4a 76 <unknown>

th.vmsle.vx v8, v4, a0, v0.t
# CHECK-INST: th.vmsle.vx v8, v4, a0, v0.t
# CHECK-ENCODING: [0x57,0x44,0x45,0x74]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 44 45 74 <unknown>

th.vmsle.vx v8, v4, a0
# CHECK-INST: th.vmsle.vx v8, v4, a0
# CHECK-ENCODING: [0x57,0x44,0x45,0x76]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 44 45 76 <unknown>

th.vmsle.vi v8, v4, 15, v0.t
# CHECK-INST: th.vmsle.vi v8, v4, 15, v0.t
# CHECK-ENCODING: [0x57,0xb4,0x47,0x74]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 b4 47 74 <unknown>

th.vmsle.vi v8, v4, 15
# CHECK-INST: th.vmsle.vi v8, v4, 15
# CHECK-ENCODING: [0x57,0xb4,0x47,0x76]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 b4 47 76 <unknown>

th.vmsgtu.vx v8, v4, a0, v0.t
# CHECK-INST: th.vmsgtu.vx v8, v4, a0, v0.t
# CHECK-ENCODING: [0x57,0x44,0x45,0x78]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 44 45 78 <unknown>

th.vmsgtu.vx v8, v4, a0
# CHECK-INST: th.vmsgtu.vx v8, v4, a0
# CHECK-ENCODING: [0x57,0x44,0x45,0x7a]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 44 45 7a <unknown>

th.vmsgtu.vi v8, v4, 15, v0.t
# CHECK-INST: th.vmsgtu.vi v8, v4, 15, v0.t
# CHECK-ENCODING: [0x57,0xb4,0x47,0x78]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 b4 47 78 <unknown>

th.vmsgtu.vi v8, v4, 15
# CHECK-INST: th.vmsgtu.vi v8, v4, 15
# CHECK-ENCODING: [0x57,0xb4,0x47,0x7a]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 b4 47 7a <unknown>

th.vmsgt.vx v8, v4, a0, v0.t
# CHECK-INST: th.vmsgt.vx v8, v4, a0, v0.t
# CHECK-ENCODING: [0x57,0x44,0x45,0x7c]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 44 45 7c <unknown>

th.vmsgt.vx v8, v4, a0
# CHECK-INST: th.vmsgt.vx v8, v4, a0
# CHECK-ENCODING: [0x57,0x44,0x45,0x7e]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 44 45 7e <unknown>

th.vmsgt.vi v8, v4, 15, v0.t
# CHECK-INST: th.vmsgt.vi v8, v4, 15, v0.t
# CHECK-ENCODING: [0x57,0xb4,0x47,0x7c]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 b4 47 7c <unknown>

th.vmsgt.vi v8, v4, 15
# CHECK-INST: th.vmsgt.vi v8, v4, 15
# CHECK-ENCODING: [0x57,0xb4,0x47,0x7e]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 b4 47 7e <unknown>

