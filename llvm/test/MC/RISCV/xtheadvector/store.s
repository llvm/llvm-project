# RUN: llvm-mc -triple=riscv64 -show-encoding --mattr=+xtheadvector %s \
# RUN:   --riscv-no-aliases | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: not llvm-mc -triple=riscv64 -show-encoding %s 2>&1 \
# RUN:   | FileCheck %s --check-prefix=CHECK-ERROR
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+xtheadvector %s \
# RUN:   | llvm-objdump -d --mattr=+xtheadvector -M no-aliases - \
# RUN:   | FileCheck %s --check-prefix=CHECK-INST
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+xtheadvector %s \
# RUN:   | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

th.vsb.v v8, (a0), v0.t
# CHECK-INST: th.vsb.v v8, (a0), v0.t
# CHECK-ENCODING: [0x27,0x04,0x05,0x00]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 04 05 00 <unknown>

th.vsb.v v8, (a0)
# CHECK-INST: th.vsb.v v8, (a0)
# CHECK-ENCODING: [0x27,0x04,0x05,0x02]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 04 05 02 <unknown>

th.vsh.v v8, (a0), v0.t
# CHECK-INST: th.vsh.v v8, (a0), v0.t
# CHECK-ENCODING: [0x27,0x54,0x05,0x00]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 54 05 00 <unknown>

th.vsh.v v8, (a0)
# CHECK-INST: th.vsh.v v8, (a0)
# CHECK-ENCODING: [0x27,0x54,0x05,0x02]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 54 05 02 <unknown>

th.vsw.v v8, (a0), v0.t
# CHECK-INST: th.vsw.v v8, (a0), v0.t
# CHECK-ENCODING: [0x27,0x64,0x05,0x00]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 64 05 00 <unknown>

th.vsw.v v8, (a0)
# CHECK-INST: th.vsw.v v8, (a0)
# CHECK-ENCODING: [0x27,0x64,0x05,0x02]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 64 05 02 <unknown>

th.vse.v v8, (a0), v0.t
# CHECK-INST: th.vse.v v8, (a0), v0.t
# CHECK-ENCODING: [0x27,0x74,0x05,0x00]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 74 05 00 <unknown>

th.vse.v v8, (a0)
# CHECK-INST: th.vse.v v8, (a0)
# CHECK-ENCODING: [0x27,0x74,0x05,0x02]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 74 05 02 <unknown>

th.vssb.v	v8, (a0), a1
# CHECK-INST: th.vssb.v	v8, (a0), a1
# CHECK-ENCODING: [0x27,0x04,0xb5,0x0a]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 04 b5 0a <unknown>

th.vssb.v	v8, (a0), a1, v0.t
# CHECK-INST: th.vssb.v	v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x04,0xb5,0x08]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 04 b5 08 <unknown>

th.vssh.v	v8, (a0), a1
# CHECK-INST: th.vssh.v	v8, (a0), a1
# CHECK-ENCODING: [0x27,0x54,0xb5,0x0a]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 54 b5 0a <unknown>

th.vssh.v	v8, (a0), a1, v0.t
# CHECK-INST: th.vssh.v	v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x54,0xb5,0x08]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 54 b5 08 <unknown>

th.vssw.v	v8, (a0), a1
# CHECK-INST: th.vssw.v	v8, (a0), a1
# CHECK-ENCODING: [0x27,0x64,0xb5,0x0a]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 64 b5 0a <unknown>

th.vssw.v	v8, (a0), a1, v0.t
# CHECK-INST: th.vssw.v	v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x64,0xb5,0x08]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 64 b5 08 <unknown>

th.vsse.v	v8, (a0), a1
# CHECK-INST: th.vsse.v	v8, (a0), a1
# CHECK-ENCODING: [0x27,0x74,0xb5,0x0a]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 74 b5 0a <unknown>

th.vsse.v	v8, (a0), a1, v0.t
# CHECK-INST: th.vsse.v	v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x74,0xb5,0x08]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 74 b5 08 <unknown>

th.vsxb.v	v8, (a0), v4
# CHECK-INST: th.vsxb.v	v8, (a0), v4
# CHECK-ENCODING: [0x27,0x04,0x45,0x0e]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 04 45 0e <unknown>

th.vsxb.v	v8, (a0), v4, v0.t
# CHECK-INST: th.vsxb.v	v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x04,0x45,0x0c]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 04 45 0c <unknown>

th.vsxh.v	v8, (a0), v4
# CHECK-INST: th.vsxh.v	v8, (a0), v4
# CHECK-ENCODING: [0x27,0x54,0x45,0x0e]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 54 45 0e <unknown>

th.vsxh.v	v8, (a0), v4, v0.t
# CHECK-INST: th.vsxh.v	v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x54,0x45,0x0c]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 54 45 0c <unknown>

th.vsxw.v	v8, (a0), v4
# CHECK-INST: th.vsxw.v	v8, (a0), v4
# CHECK-ENCODING: [0x27,0x64,0x45,0x0e]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 64 45 0e <unknown>

th.vsxw.v	v8, (a0), v4, v0.t
# CHECK-INST: th.vsxw.v	v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x64,0x45,0x0c]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 64 45 0c <unknown>

th.vsxe.v	v8, (a0), v4
# CHECK-INST: th.vsxe.v	v8, (a0), v4
# CHECK-ENCODING: [0x27,0x74,0x45,0x0e]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 74 45 0e <unknown>

th.vsxe.v	v8, (a0), v4, v0.t
# CHECK-INST: th.vsxe.v	v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x74,0x45,0x0c]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 74 45 0c <unknown>

th.vsuxb.v	v8, (a0), v4
# CHECK-INST: th.vsuxb.v	v8, (a0), v4
# CHECK-ENCODING: [0x27,0x04,0x45,0x1e]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 04 45 1e <unknown>

th.vsuxb.v	v8, (a0), v4, v0.t
# CHECK-INST: th.vsuxb.v	v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x04,0x45,0x1c]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 04 45 1c <unknown>

th.vsuxh.v	v8, (a0), v4
# CHECK-INST: th.vsuxh.v	v8, (a0), v4
# CHECK-ENCODING: [0x27,0x54,0x45,0x1e]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 54 45 1e <unknown>

th.vsuxh.v	v8, (a0), v4, v0.t
# CHECK-INST: th.vsuxh.v	v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x54,0x45,0x1c]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 54 45 1c <unknown>

th.vsuxw.v	v8, (a0), v4
# CHECK-INST: th.vsuxw.v	v8, (a0), v4
# CHECK-ENCODING: [0x27,0x64,0x45,0x1e]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 64 45 1e <unknown>

th.vsuxw.v	v8, (a0), v4, v0.t
# CHECK-INST: th.vsuxw.v	v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x64,0x45,0x1c]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 64 45 1c <unknown>

th.vsuxe.v	v8, (a0), v4
# CHECK-INST: th.vsuxe.v	v8, (a0), v4
# CHECK-ENCODING: [0x27,0x74,0x45,0x1e]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 74 45 1e <unknown>

th.vsuxe.v	v8, (a0), v4, v0.t
# CHECK-INST: th.vsuxe.v	v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x74,0x45,0x1c]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 74 45 1c <unknown>
