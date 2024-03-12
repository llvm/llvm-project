# RUN: llvm-mc -triple=riscv64 -show-encoding --mattr=+xtheadvector %s \
# RUN:   --riscv-no-aliases | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: not llvm-mc -triple=riscv64 -show-encoding %s 2>&1 \
# RUN:   | FileCheck %s --check-prefix=CHECK-ERROR
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+xtheadvector %s \
# RUN:   | llvm-objdump -d --mattr=+xtheadvector -M no-aliases - \
# RUN:   | FileCheck %s --check-prefix=CHECK-INST
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+xtheadvector %s \
# RUN:   | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

th.vlb.v v8, (a0), v0.t
# CHECK-INST: th.vlb.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x04,0x05,0x10]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 04 05 10 <unknown>

th.vlb.v v8, (a0)
# CHECK-INST: th.vlb.v v8, (a0)
# CHECK-ENCODING: [0x07,0x04,0x05,0x12]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 04 05 12 <unknown>

th.vlh.v v8, (a0), v0.t
# CHECK-INST: th.vlh.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x54,0x05,0x10]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 54 05 10 <unknown>

th.vlh.v v8, (a0)
# CHECK-INST: th.vlh.v v8, (a0)
# CHECK-ENCODING: [0x07,0x54,0x05,0x12]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 54 05 12 <unknown>

th.vlw.v v8, (a0), v0.t
# CHECK-INST: th.vlw.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x64,0x05,0x10]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 64 05 10 <unknown>

th.vlw.v v8, (a0)
# CHECK-INST: th.vlw.v v8, (a0)
# CHECK-ENCODING: [0x07,0x64,0x05,0x12]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 64 05 12 <unknown>

th.vlbu.v v8, (a0), v0.t
# CHECK-INST: th.vlbu.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x04,0x05,0x00]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 04 05 00 <unknown>

th.vlbu.v v8, (a0)
# CHECK-INST: th.vlbu.v v8, (a0)
# CHECK-ENCODING: [0x07,0x04,0x05,0x02]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 04 05 02 <unknown>

th.vlhu.v v8, (a0), v0.t
# CHECK-INST: th.vlhu.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x54,0x05,0x00]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 54 05 00 <unknown>

th.vlhu.v v8, (a0)
# CHECK-INST: th.vlhu.v v8, (a0)
# CHECK-ENCODING: [0x07,0x54,0x05,0x02]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 54 05 02 <unknown>

th.vlwu.v v8, (a0), v0.t
# CHECK-INST: th.vlwu.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x64,0x05,0x00]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 64 05 00 <unknown>

th.vlwu.v v8, (a0)
# CHECK-INST: th.vlwu.v v8, (a0)
# CHECK-ENCODING: [0x07,0x64,0x05,0x02]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 64 05 02 <unknown>

th.vle.v v8, (a0), v0.t
# CHECK-INST: th.vle.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x74,0x05,0x00]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 74 05 00 <unknown>

th.vle.v v8, (a0)
# CHECK-INST: th.vle.v v8, (a0)
# CHECK-ENCODING: [0x07,0x74,0x05,0x02]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 74 05 02 <unknown>

th.vlsb.v v8, (a0), a1, v0.t
# CHECK-INST: th.vlsb.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x04,0xb5,0x18]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 04 b5 18 <unknown>

th.vlsb.v v8, (a0), a1
# CHECK-INST: th.vlsb.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x04,0xb5,0x1a]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 04 b5 1a <unknown>

th.vlsh.v v8, (a0), a1, v0.t
# CHECK-INST: th.vlsh.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x54,0xb5,0x18]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 54 b5 18 <unknown>

th.vlsh.v v8, (a0), a1
# CHECK-INST: th.vlsh.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x54,0xb5,0x1a]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 54 b5 1a <unknown>

th.vlsw.v v8, (a0), a1, v0.t
# CHECK-INST: th.vlsw.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x64,0xb5,0x18]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 64 b5 18 <unknown>

th.vlsw.v v8, (a0), a1
# CHECK-INST: th.vlsw.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x64,0xb5,0x1a]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 64 b5 1a <unknown>

th.vlsbu.v v8, (a0), a1, v0.t
# CHECK-INST: th.vlsbu.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x04,0xb5,0x08]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 04 b5 08 <unknown>

th.vlsbu.v v8, (a0), a1
# CHECK-INST: th.vlsbu.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x04,0xb5,0x0a]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 04 b5 0a <unknown>

th.vlshu.v v8, (a0), a1, v0.t
# CHECK-INST: th.vlshu.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x54,0xb5,0x08]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 54 b5 08 <unknown>

th.vlshu.v v8, (a0), a1
# CHECK-INST: th.vlshu.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x54,0xb5,0x0a]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 54 b5 0a <unknown>

th.vlswu.v v8, (a0), a1, v0.t
# CHECK-INST: th.vlswu.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x64,0xb5,0x08]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 64 b5 08 <unknown>

th.vlswu.v v8, (a0), a1
# CHECK-INST: th.vlswu.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x64,0xb5,0x0a]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 64 b5 0a <unknown>

th.vlse.v v8, (a0), a1, v0.t
# CHECK-INST: th.vlse.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x74,0xb5,0x08]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 74 b5 08 <unknown>

th.vlse.v v8, (a0), a1
# CHECK-INST: th.vlse.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x74,0xb5,0x0a]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 74 b5 0a <unknown>

th.vlxb.v v8, (a0), v4, v0.t
# CHECK-INST: th.vlxb.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x04,0x45,0x1c]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 04 45 1c <unknown>

th.vlxb.v v8, (a0), v4
# CHECK-INST: th.vlxb.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x04,0x45,0x1e]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 04 45 1e <unknown>

th.vlxh.v v8, (a0), v4, v0.t
# CHECK-INST: th.vlxh.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x54,0x45,0x1c]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 54 45 1c <unknown>

th.vlxh.v v8, (a0), v4
# CHECK-INST: th.vlxh.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x54,0x45,0x1e]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 54 45 1e <unknown>

th.vlxw.v v8, (a0), v4, v0.t
# CHECK-INST: th.vlxw.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x64,0x45,0x1c]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 64 45 1c <unknown>

th.vlxw.v v8, (a0), v4
# CHECK-INST: th.vlxw.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x64,0x45,0x1e]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 64 45 1e <unknown>

th.vlxbu.v v8, (a0), v4, v0.t
# CHECK-INST: th.vlxbu.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x04,0x45,0x0c]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 04 45 0c <unknown>

th.vlxbu.v v8, (a0), v4
# CHECK-INST: th.vlxbu.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x04,0x45,0x0e]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 04 45 0e <unknown>

th.vlxhu.v v8, (a0), v4, v0.t
# CHECK-INST: th.vlxhu.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x54,0x45,0x0c]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 54 45 0c <unknown>

th.vlxhu.v v8, (a0), v4
# CHECK-INST: th.vlxhu.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x54,0x45,0x0e]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 54 45 0e <unknown>

th.vlxwu.v v8, (a0), v4, v0.t
# CHECK-INST: th.vlxwu.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x64,0x45,0x0c]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 64 45 0c <unknown>

th.vlxwu.v v8, (a0), v4
# CHECK-INST: th.vlxwu.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x64,0x45,0x0e]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 64 45 0e <unknown>

th.vlxe.v v8, (a0), v4, v0.t
# CHECK-INST: th.vlxe.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x74,0x45,0x0c]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 74 45 0c <unknown>

th.vlxe.v v8, (a0), v4
# CHECK-INST: th.vlxe.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x74,0x45,0x0e]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 74 45 0e <unknown>

th.vlbff.v	v8, (a0)
# CHECK-INST: th.vlbff.v	v8, (a0)
# CHECK-ENCODING: [0x07,0x04,0x05,0x13]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 04 05 13 <unknown>

th.vlbff.v	v8, (a0), v0.t
# CHECK-INST: th.vlbff.v	v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x04,0x05,0x11]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 04 05 11 <unknown>

th.vlhff.v	v8, (a0)
# CHECK-INST: th.vlhff.v	v8, (a0)
# CHECK-ENCODING: [0x07,0x54,0x05,0x13]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 54 05 13 <unknown>

th.vlhff.v	v8, (a0), v0.t
# CHECK-INST: th.vlhff.v	v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x54,0x05,0x11]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 54 05 11 <unknown>

th.vlwff.v	v8, (a0)
# CHECK-INST: th.vlwff.v	v8, (a0)
# CHECK-ENCODING: [0x07,0x64,0x05,0x13]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 64 05 13 <unknown>

th.vlwff.v	v8, (a0), v0.t
# CHECK-INST: th.vlwff.v	v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x64,0x05,0x11]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 64 05 11 <unknown>

th.vlbuff.v v8, (a0)
# CHECK-INST: th.vlbuff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x04,0x05,0x03]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 04 05 03 <unknown>

th.vlbuff.v v8, (a0), v0.t
# CHECK-INST: th.vlbuff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x04,0x05,0x01]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 04 05 01 <unknown>

th.vlhuff.v v8, (a0)
# CHECK-INST: th.vlhuff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x54,0x05,0x03]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 54 05 03 <unknown>

th.vlhuff.v v8, (a0), v0.t
# CHECK-INST: th.vlhuff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x54,0x05,0x01]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 54 05 01 <unknown>

th.vlwuff.v v8, (a0)
# CHECK-INST: th.vlwuff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x64,0x05,0x03]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 64 05 03 <unknown>

th.vlwuff.v v8, (a0), v0.t
# CHECK-INST: th.vlwuff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x64,0x05,0x01]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 64 05 01 <unknown>

th.vleff.v	v8, (a0)
# CHECK-INST: th.vleff.v	v8, (a0)
# CHECK-ENCODING: [0x07,0x74,0x05,0x03]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 74 05 03 <unknown>

th.vleff.v	v8, (a0), v0.t
# CHECK-INST: th.vleff.v	v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x74,0x05,0x01]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 74 05 01 <unknown>
