# RUN: llvm-mc -triple=riscv64 -show-encoding --mattr=+xtheadvector %s \
# RUN:   --riscv-no-aliases | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: not llvm-mc -triple=riscv64 -show-encoding %s 2>&1 \
# RUN:   | FileCheck %s --check-prefix=CHECK-ERROR
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+xtheadvector %s \
# RUN:   | llvm-objdump -d --mattr=+xtheadvector --no-print-imm-hex -M no-aliases - \
# RUN:   | FileCheck %s --check-prefix=CHECK-INST
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+xtheadvector %s \
# RUN:   | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

th.vmerge.vvm v8, v4, v20, v0
# CHECK-INST: th.vmerge.vvm v8, v4, v20, v0
# CHECK-ENCODING: [0x57,0x04,0x4a,0x5c]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 04 4a 5c <unknown>

th.vmerge.vxm v8, v4, a0, v0
# CHECK-INST: th.vmerge.vxm v8, v4, a0, v0
# CHECK-ENCODING: [0x57,0x44,0x45,0x5c]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 44 45 5c <unknown>

th.vmerge.vim v8, v4, 15, v0
# CHECK-INST: th.vmerge.vim v8, v4, 15, v0
# CHECK-ENCODING: [0x57,0xb4,0x47,0x5c]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 b4 47 5c <unknown>

th.vslideup.vx v8, v4, a0, v0.t
# CHECK-INST: th.vslideup.vx v8, v4, a0, v0.t
# CHECK-ENCODING: [0x57,0x44,0x45,0x38]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 44 45 38 <unknown>

th.vslideup.vx v8, v4, a0
# CHECK-INST: th.vslideup.vx v8, v4, a0
# CHECK-ENCODING: [0x57,0x44,0x45,0x3a]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 44 45 3a <unknown>

th.vslideup.vi v8, v4, 31, v0.t
# CHECK-INST: th.vslideup.vi v8, v4, 31, v0.t
# CHECK-ENCODING: [0x57,0xb4,0x4f,0x38]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 b4 4f 38 <unknown>

th.vslideup.vi v8, v4, 31
# CHECK-INST: th.vslideup.vi v8, v4, 31
# CHECK-ENCODING: [0x57,0xb4,0x4f,0x3a]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 b4 4f 3a <unknown>

th.vslidedown.vx v8, v4, a0, v0.t
# CHECK-INST: th.vslidedown.vx v8, v4, a0, v0.t
# CHECK-ENCODING: [0x57,0x44,0x45,0x3c]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 44 45 3c <unknown>

th.vslidedown.vx v8, v4, a0
# CHECK-INST: th.vslidedown.vx v8, v4, a0
# CHECK-ENCODING: [0x57,0x44,0x45,0x3e]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 44 45 3e <unknown>

th.vslidedown.vi v8, v4, 31, v0.t
# CHECK-INST: th.vslidedown.vi v8, v4, 31, v0.t
# CHECK-ENCODING: [0x57,0xb4,0x4f,0x3c]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 b4 4f 3c <unknown>

th.vslidedown.vi v8, v4, 31
# CHECK-INST: th.vslidedown.vi v8, v4, 31
# CHECK-ENCODING: [0x57,0xb4,0x4f,0x3e]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 b4 4f 3e <unknown>

th.vslide1up.vx v8, v4, a0, v0.t
# CHECK-INST: th.vslide1up.vx v8, v4, a0, v0.t
# CHECK-ENCODING: [0x57,0x64,0x45,0x38]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 64 45 38 <unknown>

th.vslide1up.vx v8, v4, a0
# CHECK-INST: th.vslide1up.vx v8, v4, a0
# CHECK-ENCODING: [0x57,0x64,0x45,0x3a]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 64 45 3a <unknown>

th.vslide1down.vx v8, v4, a0, v0.t
# CHECK-INST: th.vslide1down.vx v8, v4, a0, v0.t
# CHECK-ENCODING: [0x57,0x64,0x45,0x3c]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 64 45 3c <unknown>

th.vslide1down.vx v8, v4, a0
# CHECK-INST: th.vslide1down.vx v8, v4, a0
# CHECK-ENCODING: [0x57,0x64,0x45,0x3e]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 64 45 3e <unknown>

th.vrgather.vv v8, v4, v20, v0.t
# CHECK-INST: th.vrgather.vv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x04,0x4a,0x30]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 04 4a 30 <unknown>

th.vrgather.vv v8, v4, v20
# CHECK-INST: th.vrgather.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x04,0x4a,0x32]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 04 4a 32 <unknown>

th.vrgather.vx v8, v4, a0, v0.t
# CHECK-INST: th.vrgather.vx v8, v4, a0, v0.t
# CHECK-ENCODING: [0x57,0x44,0x45,0x30]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 44 45 30 <unknown>

th.vrgather.vx v8, v4, a0
# CHECK-INST: th.vrgather.vx v8, v4, a0
# CHECK-ENCODING: [0x57,0x44,0x45,0x32]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 44 45 32 <unknown>

th.vrgather.vi v8, v4, 31, v0.t
# CHECK-INST: th.vrgather.vi v8, v4, 31, v0.t
# CHECK-ENCODING: [0x57,0xb4,0x4f,0x30]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 b4 4f 30 <unknown>

th.vrgather.vi v8, v4, 31
# CHECK-INST: th.vrgather.vi v8, v4, 31
# CHECK-ENCODING: [0x57,0xb4,0x4f,0x32]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 b4 4f 32 <unknown>

th.vcompress.vm v8, v4, v20
# CHECK-INST: th.vcompress.vm v8, v4, v20
# CHECK-ENCODING: [0x57,0x24,0x4a,0x5e]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 24 4a 5e <unknown>
