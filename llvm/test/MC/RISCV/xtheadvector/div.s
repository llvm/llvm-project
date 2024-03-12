# RUN: llvm-mc -triple=riscv64 -show-encoding --mattr=+xtheadvector %s \
# RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: not llvm-mc -triple=riscv64 -show-encoding %s 2>&1 \
# RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+xtheadvector %s \
# RUN:        | llvm-objdump -d --mattr=+xtheadvector - \
# RUN:        | FileCheck %s --check-prefix=CHECK-INST
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+xtheadvector %s \
# RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

th.vdivu.vv v8, v4, v20, v0.t
# CHECK-INST: th.vdivu.vv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x24,0x4a,0x80]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 24 4a 80 <unknown>

th.vdivu.vv v8, v4, v20
# CHECK-INST: th.vdivu.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x24,0x4a,0x82]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 24 4a 82 <unknown>

th.vdivu.vx v8, v4, a0, v0.t
# CHECK-INST: th.vdivu.vx v8, v4, a0, v0.t
# CHECK-ENCODING: [0x57,0x64,0x45,0x80]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 64 45 80 <unknown>

th.vdivu.vx v8, v4, a0
# CHECK-INST: th.vdivu.vx v8, v4, a0
# CHECK-ENCODING: [0x57,0x64,0x45,0x82]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 64 45 82 <unknown>

th.vdiv.vv v8, v4, v20, v0.t
# CHECK-INST: th.vdiv.vv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x24,0x4a,0x84]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 24 4a 84 <unknown>

th.vdiv.vv v8, v4, v20
# CHECK-INST: th.vdiv.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x24,0x4a,0x86]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 24 4a 86 <unknown>

th.vdiv.vx v8, v4, a0, v0.t
# CHECK-INST: th.vdiv.vx v8, v4, a0, v0.t
# CHECK-ENCODING: [0x57,0x64,0x45,0x84]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 64 45 84 <unknown>

th.vdiv.vx v8, v4, a0
# CHECK-INST: th.vdiv.vx v8, v4, a0
# CHECK-ENCODING: [0x57,0x64,0x45,0x86]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 64 45 86 <unknown>

th.vremu.vv v8, v4, v20, v0.t
# CHECK-INST: th.vremu.vv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x24,0x4a,0x88]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 24 4a 88 <unknown>

th.vremu.vv v8, v4, v20
# CHECK-INST: th.vremu.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x24,0x4a,0x8a]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 24 4a 8a <unknown>

th.vremu.vx v8, v4, a0, v0.t
# CHECK-INST: th.vremu.vx v8, v4, a0, v0.t
# CHECK-ENCODING: [0x57,0x64,0x45,0x88]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 64 45 88 <unknown>

th.vremu.vx v8, v4, a0
# CHECK-INST: th.vremu.vx v8, v4, a0
# CHECK-ENCODING: [0x57,0x64,0x45,0x8a]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 64 45 8a <unknown>

th.vrem.vv v8, v4, v20, v0.t
# CHECK-INST: th.vrem.vv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x24,0x4a,0x8c]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 24 4a 8c <unknown>

th.vrem.vv v8, v4, v20
# CHECK-INST: th.vrem.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x24,0x4a,0x8e]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 24 4a 8e <unknown>

th.vrem.vx v8, v4, a0, v0.t
# CHECK-INST: th.vrem.vx v8, v4, a0, v0.t
# CHECK-ENCODING: [0x57,0x64,0x45,0x8c]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 64 45 8c <unknown>

th.vrem.vx v8, v4, a0
# CHECK-INST: th.vrem.vx v8, v4, a0
# CHECK-ENCODING: [0x57,0x64,0x45,0x8e]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 64 45 8e <unknown>
