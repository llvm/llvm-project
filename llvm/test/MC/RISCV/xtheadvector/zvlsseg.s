# RUN: llvm-mc -triple=riscv64 -show-encoding --mattr=+xtheadvector %s \
# RUN:   --riscv-no-aliases \
# RUN:   | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: not llvm-mc -triple=riscv64 -show-encoding %s 2>&1 \
# RUN:   | FileCheck %s --check-prefix=CHECK-ERROR
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+xtheadvector %s \
# RUN:   | llvm-objdump -d --mattr=+xtheadvector -M no-aliases - \
# RUN:   | FileCheck %s --check-prefix=CHECK-INST
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+xtheadvector %s \
# RUN:   | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

th.vlseg2b.v v8, (a0), v0.t
# CHECK-INST: th.vlseg2b.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x04,0x05,0x30]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 04 05 30 <unknown>

th.vlseg2b.v v8, (a0)
# CHECK-INST: th.vlseg2b.v v8, (a0)
# CHECK-ENCODING: [0x07,0x04,0x05,0x32]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 04 05 32 <unknown>

th.vlseg2h.v v8, (a0), v0.t
# CHECK-INST: th.vlseg2h.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x54,0x05,0x30]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 54 05 30 <unknown>

th.vlseg2h.v v8, (a0)
# CHECK-INST: th.vlseg2h.v v8, (a0)
# CHECK-ENCODING: [0x07,0x54,0x05,0x32]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 54 05 32 <unknown>

th.vlseg2w.v v8, (a0), v0.t
# CHECK-INST: th.vlseg2w.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x64,0x05,0x30]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 64 05 30 <unknown>

th.vlseg2w.v v8, (a0)
# CHECK-INST: th.vlseg2w.v v8, (a0)
# CHECK-ENCODING: [0x07,0x64,0x05,0x32]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 64 05 32 <unknown>

th.vlseg2bu.v v8, (a0), v0.t
# CHECK-INST: th.vlseg2bu.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x04,0x05,0x20]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 04 05 20 <unknown>

th.vlseg2bu.v v8, (a0)
# CHECK-INST: th.vlseg2bu.v v8, (a0)
# CHECK-ENCODING: [0x07,0x04,0x05,0x22]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 04 05 22 <unknown>

th.vlseg2hu.v v8, (a0), v0.t
# CHECK-INST: th.vlseg2hu.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x54,0x05,0x20]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 54 05 20 <unknown>

th.vlseg2hu.v v8, (a0)
# CHECK-INST: th.vlseg2hu.v v8, (a0)
# CHECK-ENCODING: [0x07,0x54,0x05,0x22]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 54 05 22 <unknown>

th.vlseg2wu.v v8, (a0), v0.t
# CHECK-INST: th.vlseg2wu.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x64,0x05,0x20]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 64 05 20 <unknown>

th.vlseg2wu.v v8, (a0)
# CHECK-INST: th.vlseg2wu.v v8, (a0)
# CHECK-ENCODING: [0x07,0x64,0x05,0x22]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 64 05 22 <unknown>

th.vlseg2e.v v8, (a0), v0.t
# CHECK-INST: th.vlseg2e.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x74,0x05,0x20]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 74 05 20 <unknown>

th.vlseg2e.v v8, (a0)
# CHECK-INST: th.vlseg2e.v v8, (a0)
# CHECK-ENCODING: [0x07,0x74,0x05,0x22]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 74 05 22 <unknown>

th.vsseg2b.v v8, (a0), v0.t
# CHECK-INST: th.vsseg2b.v v8, (a0), v0.t
# CHECK-ENCODING: [0x27,0x04,0x05,0x20]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 04 05 20 <unknown>

th.vsseg2b.v v8, (a0)
# CHECK-INST: th.vsseg2b.v v8, (a0)
# CHECK-ENCODING: [0x27,0x04,0x05,0x22]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 04 05 22 <unknown>

th.vsseg2h.v v8, (a0), v0.t
# CHECK-INST: th.vsseg2h.v v8, (a0), v0.t
# CHECK-ENCODING: [0x27,0x54,0x05,0x20]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 54 05 20 <unknown>

th.vsseg2h.v v8, (a0)
# CHECK-INST: th.vsseg2h.v v8, (a0)
# CHECK-ENCODING: [0x27,0x54,0x05,0x22]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 54 05 22 <unknown>

th.vsseg2w.v v8, (a0), v0.t
# CHECK-INST: th.vsseg2w.v v8, (a0), v0.t
# CHECK-ENCODING: [0x27,0x64,0x05,0x20]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 64 05 20 <unknown>

th.vsseg2w.v v8, (a0)
# CHECK-INST: th.vsseg2w.v v8, (a0)
# CHECK-ENCODING: [0x27,0x64,0x05,0x22]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 64 05 22 <unknown>

th.vsseg2e.v v8, (a0), v0.t
# CHECK-INST: th.vsseg2e.v v8, (a0), v0.t
# CHECK-ENCODING: [0x27,0x74,0x05,0x20]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 74 05 20 <unknown>

th.vsseg2e.v v8, (a0)
# CHECK-INST: th.vsseg2e.v v8, (a0)
# CHECK-ENCODING: [0x27,0x74,0x05,0x22]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 74 05 22 <unknown>

th.vlseg2bff.v	v8, (a0)
# CHECK-INST: th.vlseg2bff.v	v8, (a0)
# CHECK-ENCODING: [0x07,0x04,0x05,0x33]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 04 05 33 <unknown>

th.vlseg2bff.v	v8, (a0), v0.t
# CHECK-INST: th.vlseg2bff.v	v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x04,0x05,0x31]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 04 05 31 <unknown>

th.vlseg2hff.v	v8, (a0)
# CHECK-INST: th.vlseg2hff.v	v8, (a0)
# CHECK-ENCODING: [0x07,0x54,0x05,0x33]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 54 05 33 <unknown>

th.vlseg2hff.v	v8, (a0), v0.t
# CHECK-INST: th.vlseg2hff.v	v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x54,0x05,0x31]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 54 05 31 <unknown>

th.vlseg2wff.v	v8, (a0)
# CHECK-INST: th.vlseg2wff.v	v8, (a0)
# CHECK-ENCODING: [0x07,0x64,0x05,0x33]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 64 05 33 <unknown>

th.vlseg2wff.v	v8, (a0), v0.t
# CHECK-INST: th.vlseg2wff.v	v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x64,0x05,0x31]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 64 05 31 <unknown>

th.vlseg2buff.v v8, (a0)
# CHECK-INST: th.vlseg2buff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x04,0x05,0x23]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 04 05 23 <unknown>

th.vlseg2buff.v v8, (a0), v0.t
# CHECK-INST: th.vlseg2buff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x04,0x05,0x21]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 04 05 21 <unknown>

th.vlseg2huff.v v8, (a0)
# CHECK-INST: th.vlseg2huff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x54,0x05,0x23]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 54 05 23 <unknown>

th.vlseg2huff.v v8, (a0), v0.t
# CHECK-INST: th.vlseg2huff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x54,0x05,0x21]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 54 05 21 <unknown>

th.vlseg2wuff.v v8, (a0)
# CHECK-INST: th.vlseg2wuff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x64,0x05,0x23]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 64 05 23 <unknown>

th.vlseg2wuff.v v8, (a0), v0.t
# CHECK-INST: th.vlseg2wuff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x64,0x05,0x21]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 64 05 21 <unknown>

th.vlseg2eff.v	v8, (a0)
# CHECK-INST: th.vlseg2eff.v	v8, (a0)
# CHECK-ENCODING: [0x07,0x74,0x05,0x23]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 74 05 23 <unknown>

th.vlseg2eff.v	v8, (a0), v0.t
# CHECK-INST: th.vlseg2eff.v	v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x74,0x05,0x21]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 74 05 21 <unknown>

th.vlsseg2b.v v8, (a0), a1, v0.t
# CHECK-INST: th.vlsseg2b.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x04,0xb5,0x38]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 04 b5 38 <unknown>

th.vlsseg2b.v v8, (a0), a1
# CHECK-INST: th.vlsseg2b.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x04,0xb5,0x3a]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 04 b5 3a <unknown>

th.vlsseg2h.v v8, (a0), a1, v0.t
# CHECK-INST: th.vlsseg2h.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x54,0xb5,0x38]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 54 b5 38 <unknown>

th.vlsseg2h.v v8, (a0), a1
# CHECK-INST: th.vlsseg2h.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x54,0xb5,0x3a]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 54 b5 3a <unknown>

th.vlsseg2w.v v8, (a0), a1, v0.t
# CHECK-INST: th.vlsseg2w.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x64,0xb5,0x38]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 64 b5 38 <unknown>

th.vlsseg2w.v v8, (a0), a1
# CHECK-INST: th.vlsseg2w.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x64,0xb5,0x3a]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 64 b5 3a <unknown>

th.vlsseg2bu.v v8, (a0), a1, v0.t
# CHECK-INST: th.vlsseg2bu.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x04,0xb5,0x28]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 04 b5 28 <unknown>

th.vlsseg2bu.v v8, (a0), a1
# CHECK-INST: th.vlsseg2bu.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x04,0xb5,0x2a]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 04 b5 2a <unknown>

th.vlsseg2hu.v v8, (a0), a1, v0.t
# CHECK-INST: th.vlsseg2hu.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x54,0xb5,0x28]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 54 b5 28 <unknown>

th.vlsseg2hu.v v8, (a0), a1
# CHECK-INST: th.vlsseg2hu.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x54,0xb5,0x2a]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 54 b5 2a <unknown>

th.vlsseg2wu.v v8, (a0), a1, v0.t
# CHECK-INST: th.vlsseg2wu.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x64,0xb5,0x28]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 64 b5 28 <unknown>

th.vlsseg2wu.v v8, (a0), a1
# CHECK-INST: th.vlsseg2wu.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x64,0xb5,0x2a]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 64 b5 2a <unknown>

th.vlsseg2e.v v8, (a0), a1, v0.t
# CHECK-INST: th.vlsseg2e.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x74,0xb5,0x28]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 74 b5 28 <unknown>

th.vlsseg2e.v v8, (a0), a1
# CHECK-INST: th.vlsseg2e.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x74,0xb5,0x2a]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 74 b5 2a <unknown>

th.vssseg2b.v	v8, (a0), a1
# CHECK-INST: th.vssseg2b.v	v8, (a0), a1
# CHECK-ENCODING: [0x27,0x04,0xb5,0x2a]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 04 b5 2a <unknown>

th.vssseg2b.v	v8, (a0), a1, v0.t
# CHECK-INST: th.vssseg2b.v	v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x04,0xb5,0x28]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 04 b5 28 <unknown>

th.vssseg2h.v	v8, (a0), a1
# CHECK-INST: th.vssseg2h.v	v8, (a0), a1
# CHECK-ENCODING: [0x27,0x54,0xb5,0x2a]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 54 b5 2a <unknown>

th.vssseg2h.v	v8, (a0), a1, v0.t
# CHECK-INST: th.vssseg2h.v	v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x54,0xb5,0x28]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 54 b5 28 <unknown>

th.vssseg2w.v	v8, (a0), a1
# CHECK-INST: th.vssseg2w.v	v8, (a0), a1
# CHECK-ENCODING: [0x27,0x64,0xb5,0x2a]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 64 b5 2a <unknown>

th.vssseg2w.v	v8, (a0), a1, v0.t
# CHECK-INST: th.vssseg2w.v	v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x64,0xb5,0x28]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 64 b5 28 <unknown>

th.vssseg2e.v	v8, (a0), a1
# CHECK-INST: th.vssseg2e.v	v8, (a0), a1
# CHECK-ENCODING: [0x27,0x74,0xb5,0x2a]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 74 b5 2a <unknown>

th.vssseg2e.v	v8, (a0), a1, v0.t
# CHECK-INST: th.vssseg2e.v	v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x74,0xb5,0x28]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 74 b5 28 <unknown>

th.vlxseg2b.v v8, (a0), v4, v0.t
# CHECK-INST: th.vlxseg2b.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x04,0x45,0x3c]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 04 45 3c <unknown>

th.vlxseg2b.v v8, (a0), v4
# CHECK-INST: th.vlxseg2b.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x04,0x45,0x3e]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 04 45 3e <unknown>

th.vlxseg2h.v v8, (a0), v4, v0.t
# CHECK-INST: th.vlxseg2h.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x54,0x45,0x3c]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 54 45 3c <unknown>

th.vlxseg2h.v v8, (a0), v4
# CHECK-INST: th.vlxseg2h.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x54,0x45,0x3e]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 54 45 3e <unknown>

th.vlxseg2w.v v8, (a0), v4, v0.t
# CHECK-INST: th.vlxseg2w.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x64,0x45,0x3c]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 64 45 3c <unknown>

th.vlxseg2w.v v8, (a0), v4
# CHECK-INST: th.vlxseg2w.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x64,0x45,0x3e]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 64 45 3e <unknown>

th.vlxseg2bu.v v8, (a0), v4, v0.t
# CHECK-INST: th.vlxseg2bu.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x04,0x45,0x2c]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 04 45 2c <unknown>

th.vlxseg2bu.v v8, (a0), v4
# CHECK-INST: th.vlxseg2bu.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x04,0x45,0x2e]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 04 45 2e <unknown>

th.vlxseg2hu.v v8, (a0), v4, v0.t
# CHECK-INST: th.vlxseg2hu.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x54,0x45,0x2c]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 54 45 2c <unknown>

th.vlxseg2hu.v v8, (a0), v4
# CHECK-INST: th.vlxseg2hu.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x54,0x45,0x2e]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 54 45 2e <unknown>

th.vlxseg2wu.v v8, (a0), v4, v0.t
# CHECK-INST: th.vlxseg2wu.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x64,0x45,0x2c]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 64 45 2c <unknown>

th.vlxseg2wu.v v8, (a0), v4
# CHECK-INST: th.vlxseg2wu.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x64,0x45,0x2e]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 64 45 2e <unknown>

th.vlxseg2e.v v8, (a0), v4, v0.t
# CHECK-INST: th.vlxseg2e.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x74,0x45,0x2c]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 74 45 2c <unknown>

th.vlxseg2e.v v8, (a0), v4
# CHECK-INST: th.vlxseg2e.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x74,0x45,0x2e]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 74 45 2e <unknown>

th.vsxseg2b.v	v8, (a0), v4
# CHECK-INST: th.vsxseg2b.v	v8, (a0), v4
# CHECK-ENCODING: [0x27,0x04,0x45,0x2e]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 04 45 2e <unknown>

th.vsxseg2b.v	v8, (a0), v4, v0.t
# CHECK-INST: th.vsxseg2b.v	v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x04,0x45,0x2c]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 04 45 2c <unknown>

th.vsxseg2h.v	v8, (a0), v4
# CHECK-INST: th.vsxseg2h.v	v8, (a0), v4
# CHECK-ENCODING: [0x27,0x54,0x45,0x2e]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 54 45 2e <unknown>

th.vsxseg2h.v	v8, (a0), v4, v0.t
# CHECK-INST: th.vsxseg2h.v	v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x54,0x45,0x2c]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 54 45 2c <unknown>

th.vsxseg2w.v	v8, (a0), v4
# CHECK-INST: th.vsxseg2w.v	v8, (a0), v4
# CHECK-ENCODING: [0x27,0x64,0x45,0x2e]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 64 45 2e <unknown>

th.vsxseg2w.v	v8, (a0), v4, v0.t
# CHECK-INST: th.vsxseg2w.v	v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x64,0x45,0x2c]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 64 45 2c <unknown>

th.vsxseg2e.v	v8, (a0), v4
# CHECK-INST: th.vsxseg2e.v	v8, (a0), v4
# CHECK-ENCODING: [0x27,0x74,0x45,0x2e]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 74 45 2e <unknown>

th.vsxseg2e.v	v8, (a0), v4, v0.t
# CHECK-INST: th.vsxseg2e.v	v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x74,0x45,0x2c]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 74 45 2c <unknown>

th.vlseg3b.v v8, (a0), v0.t
# CHECK-INST: th.vlseg3b.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x04,0x05,0x50]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 04 05 50 <unknown>

th.vlseg3b.v v8, (a0)
# CHECK-INST: th.vlseg3b.v v8, (a0)
# CHECK-ENCODING: [0x07,0x04,0x05,0x52]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 04 05 52 <unknown>

th.vlseg3h.v v8, (a0), v0.t
# CHECK-INST: th.vlseg3h.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x54,0x05,0x50]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 54 05 50 <unknown>

th.vlseg3h.v v8, (a0)
# CHECK-INST: th.vlseg3h.v v8, (a0)
# CHECK-ENCODING: [0x07,0x54,0x05,0x52]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 54 05 52 <unknown>

th.vlseg3w.v v8, (a0), v0.t
# CHECK-INST: th.vlseg3w.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x64,0x05,0x50]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 64 05 50 <unknown>

th.vlseg3w.v v8, (a0)
# CHECK-INST: th.vlseg3w.v v8, (a0)
# CHECK-ENCODING: [0x07,0x64,0x05,0x52]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 64 05 52 <unknown>

th.vlseg3bu.v v8, (a0), v0.t
# CHECK-INST: th.vlseg3bu.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x04,0x05,0x40]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 04 05 40 <unknown>

th.vlseg3bu.v v8, (a0)
# CHECK-INST: th.vlseg3bu.v v8, (a0)
# CHECK-ENCODING: [0x07,0x04,0x05,0x42]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 04 05 42 <unknown>

th.vlseg3hu.v v8, (a0), v0.t
# CHECK-INST: th.vlseg3hu.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x54,0x05,0x40]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 54 05 40 <unknown>

th.vlseg3hu.v v8, (a0)
# CHECK-INST: th.vlseg3hu.v v8, (a0)
# CHECK-ENCODING: [0x07,0x54,0x05,0x42]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 54 05 42 <unknown>

th.vlseg3wu.v v8, (a0), v0.t
# CHECK-INST: th.vlseg3wu.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x64,0x05,0x40]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 64 05 40 <unknown>

th.vlseg3wu.v v8, (a0)
# CHECK-INST: th.vlseg3wu.v v8, (a0)
# CHECK-ENCODING: [0x07,0x64,0x05,0x42]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 64 05 42 <unknown>

th.vlseg3e.v v8, (a0), v0.t
# CHECK-INST: th.vlseg3e.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x74,0x05,0x40]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 74 05 40 <unknown>

th.vlseg3e.v v8, (a0)
# CHECK-INST: th.vlseg3e.v v8, (a0)
# CHECK-ENCODING: [0x07,0x74,0x05,0x42]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 74 05 42 <unknown>

th.vsseg3b.v v8, (a0), v0.t
# CHECK-INST: th.vsseg3b.v v8, (a0), v0.t
# CHECK-ENCODING: [0x27,0x04,0x05,0x40]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 04 05 40 <unknown>

th.vsseg3b.v v8, (a0)
# CHECK-INST: th.vsseg3b.v v8, (a0)
# CHECK-ENCODING: [0x27,0x04,0x05,0x42]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 04 05 42 <unknown>

th.vsseg3h.v v8, (a0), v0.t
# CHECK-INST: th.vsseg3h.v v8, (a0), v0.t
# CHECK-ENCODING: [0x27,0x54,0x05,0x40]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 54 05 40 <unknown>

th.vsseg3h.v v8, (a0)
# CHECK-INST: th.vsseg3h.v v8, (a0)
# CHECK-ENCODING: [0x27,0x54,0x05,0x42]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 54 05 42 <unknown>

th.vsseg3w.v v8, (a0), v0.t
# CHECK-INST: th.vsseg3w.v v8, (a0), v0.t
# CHECK-ENCODING: [0x27,0x64,0x05,0x40]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 64 05 40 <unknown>

th.vsseg3w.v v8, (a0)
# CHECK-INST: th.vsseg3w.v v8, (a0)
# CHECK-ENCODING: [0x27,0x64,0x05,0x42]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 64 05 42 <unknown>

th.vsseg3e.v v8, (a0), v0.t
# CHECK-INST: th.vsseg3e.v v8, (a0), v0.t
# CHECK-ENCODING: [0x27,0x74,0x05,0x40]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 74 05 40 <unknown>

th.vsseg3e.v v8, (a0)
# CHECK-INST: th.vsseg3e.v v8, (a0)
# CHECK-ENCODING: [0x27,0x74,0x05,0x42]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 74 05 42 <unknown>

th.vlseg3bff.v	v8, (a0)
# CHECK-INST: th.vlseg3bff.v	v8, (a0)
# CHECK-ENCODING: [0x07,0x04,0x05,0x53]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 04 05 53 <unknown>

th.vlseg3bff.v	v8, (a0), v0.t
# CHECK-INST: th.vlseg3bff.v	v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x04,0x05,0x51]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 04 05 51 <unknown>

th.vlseg3hff.v	v8, (a0)
# CHECK-INST: th.vlseg3hff.v	v8, (a0)
# CHECK-ENCODING: [0x07,0x54,0x05,0x53]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 54 05 53 <unknown>

th.vlseg3hff.v	v8, (a0), v0.t
# CHECK-INST: th.vlseg3hff.v	v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x54,0x05,0x51]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 54 05 51 <unknown>

th.vlseg3wff.v	v8, (a0)
# CHECK-INST: th.vlseg3wff.v	v8, (a0)
# CHECK-ENCODING: [0x07,0x64,0x05,0x53]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 64 05 53 <unknown>

th.vlseg3wff.v	v8, (a0), v0.t
# CHECK-INST: th.vlseg3wff.v	v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x64,0x05,0x51]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 64 05 51 <unknown>

th.vlseg3buff.v v8, (a0)
# CHECK-INST: th.vlseg3buff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x04,0x05,0x43]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 04 05 43 <unknown>

th.vlseg3buff.v v8, (a0), v0.t
# CHECK-INST: th.vlseg3buff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x04,0x05,0x41]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 04 05 41 <unknown>

th.vlseg3huff.v v8, (a0)
# CHECK-INST: th.vlseg3huff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x54,0x05,0x43]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 54 05 43 <unknown>

th.vlseg3huff.v v8, (a0), v0.t
# CHECK-INST: th.vlseg3huff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x54,0x05,0x41]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 54 05 41 <unknown>

th.vlseg3wuff.v v8, (a0)
# CHECK-INST: th.vlseg3wuff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x64,0x05,0x43]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 64 05 43 <unknown>

th.vlseg3wuff.v v8, (a0), v0.t
# CHECK-INST: th.vlseg3wuff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x64,0x05,0x41]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 64 05 41 <unknown>

th.vlseg3eff.v	v8, (a0)
# CHECK-INST: th.vlseg3eff.v	v8, (a0)
# CHECK-ENCODING: [0x07,0x74,0x05,0x43]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 74 05 43 <unknown>

th.vlseg3eff.v	v8, (a0), v0.t
# CHECK-INST: th.vlseg3eff.v	v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x74,0x05,0x41]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 74 05 41 <unknown>

th.vlsseg3b.v v8, (a0), a1, v0.t
# CHECK-INST: th.vlsseg3b.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x04,0xb5,0x58]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 04 b5 58 <unknown>

th.vlsseg3b.v v8, (a0), a1
# CHECK-INST: th.vlsseg3b.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x04,0xb5,0x5a]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 04 b5 5a <unknown>

th.vlsseg3h.v v8, (a0), a1, v0.t
# CHECK-INST: th.vlsseg3h.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x54,0xb5,0x58]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 54 b5 58 <unknown>

th.vlsseg3h.v v8, (a0), a1
# CHECK-INST: th.vlsseg3h.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x54,0xb5,0x5a]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 54 b5 5a <unknown>

th.vlsseg3w.v v8, (a0), a1, v0.t
# CHECK-INST: th.vlsseg3w.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x64,0xb5,0x58]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 64 b5 58 <unknown>

th.vlsseg3w.v v8, (a0), a1
# CHECK-INST: th.vlsseg3w.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x64,0xb5,0x5a]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 64 b5 5a <unknown>

th.vlsseg3bu.v v8, (a0), a1, v0.t
# CHECK-INST: th.vlsseg3bu.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x04,0xb5,0x48]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 04 b5 48 <unknown>

th.vlsseg3bu.v v8, (a0), a1
# CHECK-INST: th.vlsseg3bu.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x04,0xb5,0x4a]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 04 b5 4a <unknown>

th.vlsseg3hu.v v8, (a0), a1, v0.t
# CHECK-INST: th.vlsseg3hu.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x54,0xb5,0x48]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 54 b5 48 <unknown>

th.vlsseg3hu.v v8, (a0), a1
# CHECK-INST: th.vlsseg3hu.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x54,0xb5,0x4a]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 54 b5 4a <unknown>

th.vlsseg3wu.v v8, (a0), a1, v0.t
# CHECK-INST: th.vlsseg3wu.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x64,0xb5,0x48]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 64 b5 48 <unknown>

th.vlsseg3wu.v v8, (a0), a1
# CHECK-INST: th.vlsseg3wu.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x64,0xb5,0x4a]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 64 b5 4a <unknown>

th.vlsseg3e.v v8, (a0), a1, v0.t
# CHECK-INST: th.vlsseg3e.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x74,0xb5,0x48]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 74 b5 48 <unknown>

th.vlsseg3e.v v8, (a0), a1
# CHECK-INST: th.vlsseg3e.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x74,0xb5,0x4a]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 74 b5 4a <unknown>

th.vssseg3b.v	v8, (a0), a1
# CHECK-INST: th.vssseg3b.v	v8, (a0), a1
# CHECK-ENCODING: [0x27,0x04,0xb5,0x4a]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 04 b5 4a <unknown>

th.vssseg3b.v	v8, (a0), a1, v0.t
# CHECK-INST: th.vssseg3b.v	v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x04,0xb5,0x48]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 04 b5 48 <unknown>

th.vssseg3h.v	v8, (a0), a1
# CHECK-INST: th.vssseg3h.v	v8, (a0), a1
# CHECK-ENCODING: [0x27,0x54,0xb5,0x4a]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 54 b5 4a <unknown>

th.vssseg3h.v	v8, (a0), a1, v0.t
# CHECK-INST: th.vssseg3h.v	v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x54,0xb5,0x48]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 54 b5 48 <unknown>

th.vssseg3w.v	v8, (a0), a1
# CHECK-INST: th.vssseg3w.v	v8, (a0), a1
# CHECK-ENCODING: [0x27,0x64,0xb5,0x4a]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 64 b5 4a <unknown>

th.vssseg3w.v	v8, (a0), a1, v0.t
# CHECK-INST: th.vssseg3w.v	v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x64,0xb5,0x48]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 64 b5 48 <unknown>

th.vssseg3e.v	v8, (a0), a1
# CHECK-INST: th.vssseg3e.v	v8, (a0), a1
# CHECK-ENCODING: [0x27,0x74,0xb5,0x4a]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 74 b5 4a <unknown>

th.vssseg3e.v	v8, (a0), a1, v0.t
# CHECK-INST: th.vssseg3e.v	v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x74,0xb5,0x48]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 74 b5 48 <unknown>

th.vlxseg3b.v v8, (a0), v4, v0.t
# CHECK-INST: th.vlxseg3b.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x04,0x45,0x5c]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 04 45 5c <unknown>

th.vlxseg3b.v v8, (a0), v4
# CHECK-INST: th.vlxseg3b.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x04,0x45,0x5e]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 04 45 5e <unknown>

th.vlxseg3h.v v8, (a0), v4, v0.t
# CHECK-INST: th.vlxseg3h.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x54,0x45,0x5c]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 54 45 5c <unknown>

th.vlxseg3h.v v8, (a0), v4
# CHECK-INST: th.vlxseg3h.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x54,0x45,0x5e]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 54 45 5e <unknown>

th.vlxseg3w.v v8, (a0), v4, v0.t
# CHECK-INST: th.vlxseg3w.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x64,0x45,0x5c]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 64 45 5c <unknown>

th.vlxseg3w.v v8, (a0), v4
# CHECK-INST: th.vlxseg3w.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x64,0x45,0x5e]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 64 45 5e <unknown>

th.vlxseg3bu.v v8, (a0), v4, v0.t
# CHECK-INST: th.vlxseg3bu.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x04,0x45,0x4c]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 04 45 4c <unknown>

th.vlxseg3bu.v v8, (a0), v4
# CHECK-INST: th.vlxseg3bu.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x04,0x45,0x4e]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 04 45 4e <unknown>

th.vlxseg3hu.v v8, (a0), v4, v0.t
# CHECK-INST: th.vlxseg3hu.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x54,0x45,0x4c]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 54 45 4c <unknown>

th.vlxseg3hu.v v8, (a0), v4
# CHECK-INST: th.vlxseg3hu.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x54,0x45,0x4e]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 54 45 4e <unknown>

th.vlxseg3wu.v v8, (a0), v4, v0.t
# CHECK-INST: th.vlxseg3wu.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x64,0x45,0x4c]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 64 45 4c <unknown>

th.vlxseg3wu.v v8, (a0), v4
# CHECK-INST: th.vlxseg3wu.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x64,0x45,0x4e]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 64 45 4e <unknown>

th.vlxseg3e.v v8, (a0), v4, v0.t
# CHECK-INST: th.vlxseg3e.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x74,0x45,0x4c]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 74 45 4c <unknown>

th.vlxseg3e.v v8, (a0), v4
# CHECK-INST: th.vlxseg3e.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x74,0x45,0x4e]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 74 45 4e <unknown>

th.vsxseg3b.v	v8, (a0), v4
# CHECK-INST: th.vsxseg3b.v	v8, (a0), v4
# CHECK-ENCODING: [0x27,0x04,0x45,0x4e]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 04 45 4e <unknown>

th.vsxseg3b.v	v8, (a0), v4, v0.t
# CHECK-INST: th.vsxseg3b.v	v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x04,0x45,0x4c]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 04 45 4c <unknown>

th.vsxseg3h.v	v8, (a0), v4
# CHECK-INST: th.vsxseg3h.v	v8, (a0), v4
# CHECK-ENCODING: [0x27,0x54,0x45,0x4e]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 54 45 4e <unknown>

th.vsxseg3h.v	v8, (a0), v4, v0.t
# CHECK-INST: th.vsxseg3h.v	v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x54,0x45,0x4c]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 54 45 4c <unknown>

th.vsxseg3w.v	v8, (a0), v4
# CHECK-INST: th.vsxseg3w.v	v8, (a0), v4
# CHECK-ENCODING: [0x27,0x64,0x45,0x4e]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 64 45 4e <unknown>

th.vsxseg3w.v	v8, (a0), v4, v0.t
# CHECK-INST: th.vsxseg3w.v	v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x64,0x45,0x4c]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 64 45 4c <unknown>

th.vsxseg3e.v	v8, (a0), v4
# CHECK-INST: th.vsxseg3e.v	v8, (a0), v4
# CHECK-ENCODING: [0x27,0x74,0x45,0x4e]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 74 45 4e <unknown>

th.vsxseg3e.v	v8, (a0), v4, v0.t
# CHECK-INST: th.vsxseg3e.v	v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x74,0x45,0x4c]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 74 45 4c <unknown>

th.vlseg4b.v v8, (a0), v0.t
# CHECK-INST: th.vlseg4b.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x04,0x05,0x70]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 04 05 70 <unknown>

th.vlseg4b.v v8, (a0)
# CHECK-INST: th.vlseg4b.v v8, (a0)
# CHECK-ENCODING: [0x07,0x04,0x05,0x72]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 04 05 72 <unknown>

th.vlseg4h.v v8, (a0), v0.t
# CHECK-INST: th.vlseg4h.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x54,0x05,0x70]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 54 05 70 <unknown>

th.vlseg4h.v v8, (a0)
# CHECK-INST: th.vlseg4h.v v8, (a0)
# CHECK-ENCODING: [0x07,0x54,0x05,0x72]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 54 05 72 <unknown>

th.vlseg4w.v v8, (a0), v0.t
# CHECK-INST: th.vlseg4w.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x64,0x05,0x70]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 64 05 70 <unknown>

th.vlseg4w.v v8, (a0)
# CHECK-INST: th.vlseg4w.v v8, (a0)
# CHECK-ENCODING: [0x07,0x64,0x05,0x72]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 64 05 72 <unknown>

th.vlseg4bu.v v8, (a0), v0.t
# CHECK-INST: th.vlseg4bu.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x04,0x05,0x60]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 04 05 60 <unknown>

th.vlseg4bu.v v8, (a0)
# CHECK-INST: th.vlseg4bu.v v8, (a0)
# CHECK-ENCODING: [0x07,0x04,0x05,0x62]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 04 05 62 <unknown>

th.vlseg4hu.v v8, (a0), v0.t
# CHECK-INST: th.vlseg4hu.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x54,0x05,0x60]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 54 05 60 <unknown>

th.vlseg4hu.v v8, (a0)
# CHECK-INST: th.vlseg4hu.v v8, (a0)
# CHECK-ENCODING: [0x07,0x54,0x05,0x62]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 54 05 62 <unknown>

th.vlseg4wu.v v8, (a0), v0.t
# CHECK-INST: th.vlseg4wu.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x64,0x05,0x60]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 64 05 60 <unknown>

th.vlseg4wu.v v8, (a0)
# CHECK-INST: th.vlseg4wu.v v8, (a0)
# CHECK-ENCODING: [0x07,0x64,0x05,0x62]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 64 05 62 <unknown>

th.vlseg4e.v v8, (a0), v0.t
# CHECK-INST: th.vlseg4e.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x74,0x05,0x60]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 74 05 60 <unknown>

th.vlseg4e.v v8, (a0)
# CHECK-INST: th.vlseg4e.v v8, (a0)
# CHECK-ENCODING: [0x07,0x74,0x05,0x62]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 74 05 62 <unknown>

th.vsseg4b.v v8, (a0), v0.t
# CHECK-INST: th.vsseg4b.v v8, (a0), v0.t
# CHECK-ENCODING: [0x27,0x04,0x05,0x60]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 04 05 60 <unknown>

th.vsseg4b.v v8, (a0)
# CHECK-INST: th.vsseg4b.v v8, (a0)
# CHECK-ENCODING: [0x27,0x04,0x05,0x62]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 04 05 62 <unknown>

th.vsseg4h.v v8, (a0), v0.t
# CHECK-INST: th.vsseg4h.v v8, (a0), v0.t
# CHECK-ENCODING: [0x27,0x54,0x05,0x60]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 54 05 60 <unknown>

th.vsseg4h.v v8, (a0)
# CHECK-INST: th.vsseg4h.v v8, (a0)
# CHECK-ENCODING: [0x27,0x54,0x05,0x62]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 54 05 62 <unknown>

th.vsseg4w.v v8, (a0), v0.t
# CHECK-INST: th.vsseg4w.v v8, (a0), v0.t
# CHECK-ENCODING: [0x27,0x64,0x05,0x60]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 64 05 60 <unknown>

th.vsseg4w.v v8, (a0)
# CHECK-INST: th.vsseg4w.v v8, (a0)
# CHECK-ENCODING: [0x27,0x64,0x05,0x62]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 64 05 62 <unknown>

th.vsseg4e.v v8, (a0), v0.t
# CHECK-INST: th.vsseg4e.v v8, (a0), v0.t
# CHECK-ENCODING: [0x27,0x74,0x05,0x60]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 74 05 60 <unknown>

th.vsseg4e.v v8, (a0)
# CHECK-INST: th.vsseg4e.v v8, (a0)
# CHECK-ENCODING: [0x27,0x74,0x05,0x62]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 74 05 62 <unknown>

th.vlseg4bff.v	v8, (a0)
# CHECK-INST: th.vlseg4bff.v	v8, (a0)
# CHECK-ENCODING: [0x07,0x04,0x05,0x73]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 04 05 73 <unknown>

th.vlseg4bff.v	v8, (a0), v0.t
# CHECK-INST: th.vlseg4bff.v	v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x04,0x05,0x71]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 04 05 71 <unknown>

th.vlseg4hff.v	v8, (a0)
# CHECK-INST: th.vlseg4hff.v	v8, (a0)
# CHECK-ENCODING: [0x07,0x54,0x05,0x73]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 54 05 73 <unknown>

th.vlseg4hff.v	v8, (a0), v0.t
# CHECK-INST: th.vlseg4hff.v	v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x54,0x05,0x71]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 54 05 71 <unknown>

th.vlseg4wff.v	v8, (a0)
# CHECK-INST: th.vlseg4wff.v	v8, (a0)
# CHECK-ENCODING: [0x07,0x64,0x05,0x73]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 64 05 73 <unknown>

th.vlseg4wff.v	v8, (a0), v0.t
# CHECK-INST: th.vlseg4wff.v	v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x64,0x05,0x71]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 64 05 71 <unknown>

th.vlseg4buff.v v8, (a0)
# CHECK-INST: th.vlseg4buff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x04,0x05,0x63]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 04 05 63 <unknown>

th.vlseg4buff.v v8, (a0), v0.t
# CHECK-INST: th.vlseg4buff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x04,0x05,0x61]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 04 05 61 <unknown>

th.vlseg4huff.v v8, (a0)
# CHECK-INST: th.vlseg4huff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x54,0x05,0x63]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 54 05 63 <unknown>

th.vlseg4huff.v v8, (a0), v0.t
# CHECK-INST: th.vlseg4huff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x54,0x05,0x61]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 54 05 61 <unknown>

th.vlseg4wuff.v v8, (a0)
# CHECK-INST: th.vlseg4wuff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x64,0x05,0x63]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 64 05 63 <unknown>

th.vlseg4wuff.v v8, (a0), v0.t
# CHECK-INST: th.vlseg4wuff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x64,0x05,0x61]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 64 05 61 <unknown>

th.vlseg4eff.v	v8, (a0)
# CHECK-INST: th.vlseg4eff.v	v8, (a0)
# CHECK-ENCODING: [0x07,0x74,0x05,0x63]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 74 05 63 <unknown>

th.vlseg4eff.v	v8, (a0), v0.t
# CHECK-INST: th.vlseg4eff.v	v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x74,0x05,0x61]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 74 05 61 <unknown>

th.vlsseg4b.v v8, (a0), a1, v0.t
# CHECK-INST: th.vlsseg4b.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x04,0xb5,0x78]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 04 b5 78 <unknown>

th.vlsseg4b.v v8, (a0), a1
# CHECK-INST: th.vlsseg4b.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x04,0xb5,0x7a]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 04 b5 7a <unknown>

th.vlsseg4h.v v8, (a0), a1, v0.t
# CHECK-INST: th.vlsseg4h.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x54,0xb5,0x78]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 54 b5 78 <unknown>

th.vlsseg4h.v v8, (a0), a1
# CHECK-INST: th.vlsseg4h.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x54,0xb5,0x7a]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 54 b5 7a <unknown>

th.vlsseg4w.v v8, (a0), a1, v0.t
# CHECK-INST: th.vlsseg4w.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x64,0xb5,0x78]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 64 b5 78 <unknown>

th.vlsseg4w.v v8, (a0), a1
# CHECK-INST: th.vlsseg4w.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x64,0xb5,0x7a]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 64 b5 7a <unknown>

th.vlsseg4bu.v v8, (a0), a1, v0.t
# CHECK-INST: th.vlsseg4bu.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x04,0xb5,0x68]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 04 b5 68 <unknown>

th.vlsseg4bu.v v8, (a0), a1
# CHECK-INST: th.vlsseg4bu.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x04,0xb5,0x6a]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 04 b5 6a <unknown>

th.vlsseg4hu.v v8, (a0), a1, v0.t
# CHECK-INST: th.vlsseg4hu.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x54,0xb5,0x68]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 54 b5 68 <unknown>

th.vlsseg4hu.v v8, (a0), a1
# CHECK-INST: th.vlsseg4hu.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x54,0xb5,0x6a]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 54 b5 6a <unknown>

th.vlsseg4wu.v v8, (a0), a1, v0.t
# CHECK-INST: th.vlsseg4wu.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x64,0xb5,0x68]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 64 b5 68 <unknown>

th.vlsseg4wu.v v8, (a0), a1
# CHECK-INST: th.vlsseg4wu.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x64,0xb5,0x6a]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 64 b5 6a <unknown>

th.vlsseg4e.v v8, (a0), a1, v0.t
# CHECK-INST: th.vlsseg4e.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x74,0xb5,0x68]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 74 b5 68 <unknown>

th.vlsseg4e.v v8, (a0), a1
# CHECK-INST: th.vlsseg4e.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x74,0xb5,0x6a]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 74 b5 6a <unknown>

th.vssseg4b.v	v8, (a0), a1
# CHECK-INST: th.vssseg4b.v	v8, (a0), a1
# CHECK-ENCODING: [0x27,0x04,0xb5,0x6a]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 04 b5 6a <unknown>

th.vssseg4b.v	v8, (a0), a1, v0.t
# CHECK-INST: th.vssseg4b.v	v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x04,0xb5,0x68]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 04 b5 68 <unknown>

th.vssseg4h.v	v8, (a0), a1
# CHECK-INST: th.vssseg4h.v	v8, (a0), a1
# CHECK-ENCODING: [0x27,0x54,0xb5,0x6a]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 54 b5 6a <unknown>

th.vssseg4h.v	v8, (a0), a1, v0.t
# CHECK-INST: th.vssseg4h.v	v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x54,0xb5,0x68]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 54 b5 68 <unknown>

th.vssseg4w.v	v8, (a0), a1
# CHECK-INST: th.vssseg4w.v	v8, (a0), a1
# CHECK-ENCODING: [0x27,0x64,0xb5,0x6a]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 64 b5 6a <unknown>

th.vssseg4w.v	v8, (a0), a1, v0.t
# CHECK-INST: th.vssseg4w.v	v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x64,0xb5,0x68]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 64 b5 68 <unknown>

th.vssseg4e.v	v8, (a0), a1
# CHECK-INST: th.vssseg4e.v	v8, (a0), a1
# CHECK-ENCODING: [0x27,0x74,0xb5,0x6a]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 74 b5 6a <unknown>

th.vssseg4e.v	v8, (a0), a1, v0.t
# CHECK-INST: th.vssseg4e.v	v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x74,0xb5,0x68]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 74 b5 68 <unknown>

th.vlxseg4b.v v8, (a0), v4, v0.t
# CHECK-INST: th.vlxseg4b.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x04,0x45,0x7c]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 04 45 7c <unknown>

th.vlxseg4b.v v8, (a0), v4
# CHECK-INST: th.vlxseg4b.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x04,0x45,0x7e]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 04 45 7e <unknown>

th.vlxseg4h.v v8, (a0), v4, v0.t
# CHECK-INST: th.vlxseg4h.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x54,0x45,0x7c]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 54 45 7c <unknown>

th.vlxseg4h.v v8, (a0), v4
# CHECK-INST: th.vlxseg4h.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x54,0x45,0x7e]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 54 45 7e <unknown>

th.vlxseg4w.v v8, (a0), v4, v0.t
# CHECK-INST: th.vlxseg4w.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x64,0x45,0x7c]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 64 45 7c <unknown>

th.vlxseg4w.v v8, (a0), v4
# CHECK-INST: th.vlxseg4w.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x64,0x45,0x7e]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 64 45 7e <unknown>

th.vlxseg4bu.v v8, (a0), v4, v0.t
# CHECK-INST: th.vlxseg4bu.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x04,0x45,0x6c]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 04 45 6c <unknown>

th.vlxseg4bu.v v8, (a0), v4
# CHECK-INST: th.vlxseg4bu.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x04,0x45,0x6e]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 04 45 6e <unknown>

th.vlxseg4hu.v v8, (a0), v4, v0.t
# CHECK-INST: th.vlxseg4hu.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x54,0x45,0x6c]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 54 45 6c <unknown>

th.vlxseg4hu.v v8, (a0), v4
# CHECK-INST: th.vlxseg4hu.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x54,0x45,0x6e]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 54 45 6e <unknown>

th.vlxseg4wu.v v8, (a0), v4, v0.t
# CHECK-INST: th.vlxseg4wu.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x64,0x45,0x6c]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 64 45 6c <unknown>

th.vlxseg4wu.v v8, (a0), v4
# CHECK-INST: th.vlxseg4wu.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x64,0x45,0x6e]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 64 45 6e <unknown>

th.vlxseg4e.v v8, (a0), v4, v0.t
# CHECK-INST: th.vlxseg4e.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x74,0x45,0x6c]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 74 45 6c <unknown>

th.vlxseg4e.v v8, (a0), v4
# CHECK-INST: th.vlxseg4e.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x74,0x45,0x6e]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 74 45 6e <unknown>

th.vsxseg4b.v	v8, (a0), v4
# CHECK-INST: th.vsxseg4b.v	v8, (a0), v4
# CHECK-ENCODING: [0x27,0x04,0x45,0x6e]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 04 45 6e <unknown>

th.vsxseg4b.v	v8, (a0), v4, v0.t
# CHECK-INST: th.vsxseg4b.v	v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x04,0x45,0x6c]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 04 45 6c <unknown>

th.vsxseg4h.v	v8, (a0), v4
# CHECK-INST: th.vsxseg4h.v	v8, (a0), v4
# CHECK-ENCODING: [0x27,0x54,0x45,0x6e]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 54 45 6e <unknown>

th.vsxseg4h.v	v8, (a0), v4, v0.t
# CHECK-INST: th.vsxseg4h.v	v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x54,0x45,0x6c]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 54 45 6c <unknown>

th.vsxseg4w.v	v8, (a0), v4
# CHECK-INST: th.vsxseg4w.v	v8, (a0), v4
# CHECK-ENCODING: [0x27,0x64,0x45,0x6e]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 64 45 6e <unknown>

th.vsxseg4w.v	v8, (a0), v4, v0.t
# CHECK-INST: th.vsxseg4w.v	v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x64,0x45,0x6c]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 64 45 6c <unknown>

th.vsxseg4e.v	v8, (a0), v4
# CHECK-INST: th.vsxseg4e.v	v8, (a0), v4
# CHECK-ENCODING: [0x27,0x74,0x45,0x6e]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 74 45 6e <unknown>

th.vsxseg4e.v	v8, (a0), v4, v0.t
# CHECK-INST: th.vsxseg4e.v	v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x74,0x45,0x6c]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 74 45 6c <unknown>

th.vlseg5b.v v8, (a0), v0.t
# CHECK-INST: th.vlseg5b.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x04,0x05,0x90]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 04 05 90 <unknown>

th.vlseg5b.v v8, (a0)
# CHECK-INST: th.vlseg5b.v v8, (a0)
# CHECK-ENCODING: [0x07,0x04,0x05,0x92]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 04 05 92 <unknown>

th.vlseg5h.v v8, (a0), v0.t
# CHECK-INST: th.vlseg5h.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x54,0x05,0x90]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 54 05 90 <unknown>

th.vlseg5h.v v8, (a0)
# CHECK-INST: th.vlseg5h.v v8, (a0)
# CHECK-ENCODING: [0x07,0x54,0x05,0x92]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 54 05 92 <unknown>

th.vlseg5w.v v8, (a0), v0.t
# CHECK-INST: th.vlseg5w.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x64,0x05,0x90]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 64 05 90 <unknown>

th.vlseg5w.v v8, (a0)
# CHECK-INST: th.vlseg5w.v v8, (a0)
# CHECK-ENCODING: [0x07,0x64,0x05,0x92]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 64 05 92 <unknown>

th.vlseg5bu.v v8, (a0), v0.t
# CHECK-INST: th.vlseg5bu.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x04,0x05,0x80]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 04 05 80 <unknown>

th.vlseg5bu.v v8, (a0)
# CHECK-INST: th.vlseg5bu.v v8, (a0)
# CHECK-ENCODING: [0x07,0x04,0x05,0x82]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 04 05 82 <unknown>

th.vlseg5hu.v v8, (a0), v0.t
# CHECK-INST: th.vlseg5hu.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x54,0x05,0x80]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 54 05 80 <unknown>

th.vlseg5hu.v v8, (a0)
# CHECK-INST: th.vlseg5hu.v v8, (a0)
# CHECK-ENCODING: [0x07,0x54,0x05,0x82]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 54 05 82 <unknown>

th.vlseg5wu.v v8, (a0), v0.t
# CHECK-INST: th.vlseg5wu.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x64,0x05,0x80]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 64 05 80 <unknown>

th.vlseg5wu.v v8, (a0)
# CHECK-INST: th.vlseg5wu.v v8, (a0)
# CHECK-ENCODING: [0x07,0x64,0x05,0x82]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 64 05 82 <unknown>

th.vlseg5e.v v8, (a0), v0.t
# CHECK-INST: th.vlseg5e.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x74,0x05,0x80]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 74 05 80 <unknown>

th.vlseg5e.v v8, (a0)
# CHECK-INST: th.vlseg5e.v v8, (a0)
# CHECK-ENCODING: [0x07,0x74,0x05,0x82]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 74 05 82 <unknown>

th.vsseg5b.v v8, (a0), v0.t
# CHECK-INST: th.vsseg5b.v v8, (a0), v0.t
# CHECK-ENCODING: [0x27,0x04,0x05,0x80]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 04 05 80 <unknown>

th.vsseg5b.v v8, (a0)
# CHECK-INST: th.vsseg5b.v v8, (a0)
# CHECK-ENCODING: [0x27,0x04,0x05,0x82]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 04 05 82 <unknown>

th.vsseg5h.v v8, (a0), v0.t
# CHECK-INST: th.vsseg5h.v v8, (a0), v0.t
# CHECK-ENCODING: [0x27,0x54,0x05,0x80]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 54 05 80 <unknown>

th.vsseg5h.v v8, (a0)
# CHECK-INST: th.vsseg5h.v v8, (a0)
# CHECK-ENCODING: [0x27,0x54,0x05,0x82]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 54 05 82 <unknown>

th.vsseg5w.v v8, (a0), v0.t
# CHECK-INST: th.vsseg5w.v v8, (a0), v0.t
# CHECK-ENCODING: [0x27,0x64,0x05,0x80]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 64 05 80 <unknown>

th.vsseg5w.v v8, (a0)
# CHECK-INST: th.vsseg5w.v v8, (a0)
# CHECK-ENCODING: [0x27,0x64,0x05,0x82]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 64 05 82 <unknown>

th.vsseg5e.v v8, (a0), v0.t
# CHECK-INST: th.vsseg5e.v v8, (a0), v0.t
# CHECK-ENCODING: [0x27,0x74,0x05,0x80]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 74 05 80 <unknown>

th.vsseg5e.v v8, (a0)
# CHECK-INST: th.vsseg5e.v v8, (a0)
# CHECK-ENCODING: [0x27,0x74,0x05,0x82]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 74 05 82 <unknown>

th.vlseg5bff.v	v8, (a0)
# CHECK-INST: th.vlseg5bff.v	v8, (a0)
# CHECK-ENCODING: [0x07,0x04,0x05,0x93]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 04 05 93 <unknown>

th.vlseg5bff.v	v8, (a0), v0.t
# CHECK-INST: th.vlseg5bff.v	v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x04,0x05,0x91]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 04 05 91 <unknown>

th.vlseg5hff.v	v8, (a0)
# CHECK-INST: th.vlseg5hff.v	v8, (a0)
# CHECK-ENCODING: [0x07,0x54,0x05,0x93]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 54 05 93 <unknown>

th.vlseg5hff.v	v8, (a0), v0.t
# CHECK-INST: th.vlseg5hff.v	v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x54,0x05,0x91]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 54 05 91 <unknown>

th.vlseg5wff.v	v8, (a0)
# CHECK-INST: th.vlseg5wff.v	v8, (a0)
# CHECK-ENCODING: [0x07,0x64,0x05,0x93]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 64 05 93 <unknown>

th.vlseg5wff.v	v8, (a0), v0.t
# CHECK-INST: th.vlseg5wff.v	v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x64,0x05,0x91]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 64 05 91 <unknown>

th.vlseg5buff.v v8, (a0)
# CHECK-INST: th.vlseg5buff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x04,0x05,0x83]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 04 05 83 <unknown>

th.vlseg5buff.v v8, (a0), v0.t
# CHECK-INST: th.vlseg5buff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x04,0x05,0x81]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 04 05 81 <unknown>

th.vlseg5huff.v v8, (a0)
# CHECK-INST: th.vlseg5huff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x54,0x05,0x83]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 54 05 83 <unknown>

th.vlseg5huff.v v8, (a0), v0.t
# CHECK-INST: th.vlseg5huff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x54,0x05,0x81]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 54 05 81 <unknown>

th.vlseg5wuff.v v8, (a0)
# CHECK-INST: th.vlseg5wuff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x64,0x05,0x83]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 64 05 83 <unknown>

th.vlseg5wuff.v v8, (a0), v0.t
# CHECK-INST: th.vlseg5wuff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x64,0x05,0x81]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 64 05 81 <unknown>

th.vlseg5eff.v	v8, (a0)
# CHECK-INST: th.vlseg5eff.v	v8, (a0)
# CHECK-ENCODING: [0x07,0x74,0x05,0x83]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 74 05 83 <unknown>

th.vlseg5eff.v	v8, (a0), v0.t
# CHECK-INST: th.vlseg5eff.v	v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x74,0x05,0x81]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 74 05 81 <unknown>

th.vlsseg5b.v v8, (a0), a1, v0.t
# CHECK-INST: th.vlsseg5b.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x04,0xb5,0x98]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 04 b5 98 <unknown>

th.vlsseg5b.v v8, (a0), a1
# CHECK-INST: th.vlsseg5b.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x04,0xb5,0x9a]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 04 b5 9a <unknown>

th.vlsseg5h.v v8, (a0), a1, v0.t
# CHECK-INST: th.vlsseg5h.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x54,0xb5,0x98]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 54 b5 98 <unknown>

th.vlsseg5h.v v8, (a0), a1
# CHECK-INST: th.vlsseg5h.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x54,0xb5,0x9a]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 54 b5 9a <unknown>

th.vlsseg5w.v v8, (a0), a1, v0.t
# CHECK-INST: th.vlsseg5w.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x64,0xb5,0x98]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 64 b5 98 <unknown>

th.vlsseg5w.v v8, (a0), a1
# CHECK-INST: th.vlsseg5w.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x64,0xb5,0x9a]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 64 b5 9a <unknown>

th.vlsseg5bu.v v8, (a0), a1, v0.t
# CHECK-INST: th.vlsseg5bu.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x04,0xb5,0x88]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 04 b5 88 <unknown>

th.vlsseg5bu.v v8, (a0), a1
# CHECK-INST: th.vlsseg5bu.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x04,0xb5,0x8a]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 04 b5 8a <unknown>

th.vlsseg5hu.v v8, (a0), a1, v0.t
# CHECK-INST: th.vlsseg5hu.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x54,0xb5,0x88]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 54 b5 88 <unknown>

th.vlsseg5hu.v v8, (a0), a1
# CHECK-INST: th.vlsseg5hu.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x54,0xb5,0x8a]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 54 b5 8a <unknown>

th.vlsseg5wu.v v8, (a0), a1, v0.t
# CHECK-INST: th.vlsseg5wu.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x64,0xb5,0x88]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 64 b5 88 <unknown>

th.vlsseg5wu.v v8, (a0), a1
# CHECK-INST: th.vlsseg5wu.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x64,0xb5,0x8a]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 64 b5 8a <unknown>

th.vlsseg5e.v v8, (a0), a1, v0.t
# CHECK-INST: th.vlsseg5e.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x74,0xb5,0x88]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 74 b5 88 <unknown>

th.vlsseg5e.v v8, (a0), a1
# CHECK-INST: th.vlsseg5e.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x74,0xb5,0x8a]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 74 b5 8a <unknown>

th.vssseg5b.v	v8, (a0), a1
# CHECK-INST: th.vssseg5b.v	v8, (a0), a1
# CHECK-ENCODING: [0x27,0x04,0xb5,0x8a]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 04 b5 8a <unknown>

th.vssseg5b.v	v8, (a0), a1, v0.t
# CHECK-INST: th.vssseg5b.v	v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x04,0xb5,0x88]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 04 b5 88 <unknown>

th.vssseg5h.v	v8, (a0), a1
# CHECK-INST: th.vssseg5h.v	v8, (a0), a1
# CHECK-ENCODING: [0x27,0x54,0xb5,0x8a]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 54 b5 8a <unknown>

th.vssseg5h.v	v8, (a0), a1, v0.t
# CHECK-INST: th.vssseg5h.v	v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x54,0xb5,0x88]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 54 b5 88 <unknown>

th.vssseg5w.v	v8, (a0), a1
# CHECK-INST: th.vssseg5w.v	v8, (a0), a1
# CHECK-ENCODING: [0x27,0x64,0xb5,0x8a]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 64 b5 8a <unknown>

th.vssseg5w.v	v8, (a0), a1, v0.t
# CHECK-INST: th.vssseg5w.v	v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x64,0xb5,0x88]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 64 b5 88 <unknown>

th.vssseg5e.v	v8, (a0), a1
# CHECK-INST: th.vssseg5e.v	v8, (a0), a1
# CHECK-ENCODING: [0x27,0x74,0xb5,0x8a]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 74 b5 8a <unknown>

th.vssseg5e.v	v8, (a0), a1, v0.t
# CHECK-INST: th.vssseg5e.v	v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x74,0xb5,0x88]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 74 b5 88 <unknown>

th.vlxseg5b.v v8, (a0), v4, v0.t
# CHECK-INST: th.vlxseg5b.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x04,0x45,0x9c]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 04 45 9c <unknown>

th.vlxseg5b.v v8, (a0), v4
# CHECK-INST: th.vlxseg5b.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x04,0x45,0x9e]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 04 45 9e <unknown>

th.vlxseg5h.v v8, (a0), v4, v0.t
# CHECK-INST: th.vlxseg5h.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x54,0x45,0x9c]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 54 45 9c <unknown>

th.vlxseg5h.v v8, (a0), v4
# CHECK-INST: th.vlxseg5h.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x54,0x45,0x9e]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 54 45 9e <unknown>

th.vlxseg5w.v v8, (a0), v4, v0.t
# CHECK-INST: th.vlxseg5w.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x64,0x45,0x9c]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 64 45 9c <unknown>

th.vlxseg5w.v v8, (a0), v4
# CHECK-INST: th.vlxseg5w.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x64,0x45,0x9e]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 64 45 9e <unknown>

th.vlxseg5bu.v v8, (a0), v4, v0.t
# CHECK-INST: th.vlxseg5bu.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x04,0x45,0x8c]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 04 45 8c <unknown>

th.vlxseg5bu.v v8, (a0), v4
# CHECK-INST: th.vlxseg5bu.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x04,0x45,0x8e]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 04 45 8e <unknown>

th.vlxseg5hu.v v8, (a0), v4, v0.t
# CHECK-INST: th.vlxseg5hu.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x54,0x45,0x8c]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 54 45 8c <unknown>

th.vlxseg5hu.v v8, (a0), v4
# CHECK-INST: th.vlxseg5hu.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x54,0x45,0x8e]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 54 45 8e <unknown>

th.vlxseg5wu.v v8, (a0), v4, v0.t
# CHECK-INST: th.vlxseg5wu.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x64,0x45,0x8c]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 64 45 8c <unknown>

th.vlxseg5wu.v v8, (a0), v4
# CHECK-INST: th.vlxseg5wu.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x64,0x45,0x8e]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 64 45 8e <unknown>

th.vlxseg5e.v v8, (a0), v4, v0.t
# CHECK-INST: th.vlxseg5e.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x74,0x45,0x8c]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 74 45 8c <unknown>

th.vlxseg5e.v v8, (a0), v4
# CHECK-INST: th.vlxseg5e.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x74,0x45,0x8e]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 74 45 8e <unknown>

th.vsxseg5b.v	v8, (a0), v4
# CHECK-INST: th.vsxseg5b.v	v8, (a0), v4
# CHECK-ENCODING: [0x27,0x04,0x45,0x8e]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 04 45 8e <unknown>

th.vsxseg5b.v	v8, (a0), v4, v0.t
# CHECK-INST: th.vsxseg5b.v	v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x04,0x45,0x8c]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 04 45 8c <unknown>

th.vsxseg5h.v	v8, (a0), v4
# CHECK-INST: th.vsxseg5h.v	v8, (a0), v4
# CHECK-ENCODING: [0x27,0x54,0x45,0x8e]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 54 45 8e <unknown>

th.vsxseg5h.v	v8, (a0), v4, v0.t
# CHECK-INST: th.vsxseg5h.v	v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x54,0x45,0x8c]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 54 45 8c <unknown>

th.vsxseg5w.v	v8, (a0), v4
# CHECK-INST: th.vsxseg5w.v	v8, (a0), v4
# CHECK-ENCODING: [0x27,0x64,0x45,0x8e]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 64 45 8e <unknown>

th.vsxseg5w.v	v8, (a0), v4, v0.t
# CHECK-INST: th.vsxseg5w.v	v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x64,0x45,0x8c]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 64 45 8c <unknown>

th.vsxseg5e.v	v8, (a0), v4
# CHECK-INST: th.vsxseg5e.v	v8, (a0), v4
# CHECK-ENCODING: [0x27,0x74,0x45,0x8e]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 74 45 8e <unknown>

th.vsxseg5e.v	v8, (a0), v4, v0.t
# CHECK-INST: th.vsxseg5e.v	v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x74,0x45,0x8c]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 74 45 8c <unknown>

th.vlseg6b.v v8, (a0), v0.t
# CHECK-INST: th.vlseg6b.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x04,0x05,0xb0]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 04 05 b0 <unknown>

th.vlseg6b.v v8, (a0)
# CHECK-INST: th.vlseg6b.v v8, (a0)
# CHECK-ENCODING: [0x07,0x04,0x05,0xb2]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 04 05 b2 <unknown>

th.vlseg6h.v v8, (a0), v0.t
# CHECK-INST: th.vlseg6h.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x54,0x05,0xb0]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 54 05 b0 <unknown>

th.vlseg6h.v v8, (a0)
# CHECK-INST: th.vlseg6h.v v8, (a0)
# CHECK-ENCODING: [0x07,0x54,0x05,0xb2]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 54 05 b2 <unknown>

th.vlseg6w.v v8, (a0), v0.t
# CHECK-INST: th.vlseg6w.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x64,0x05,0xb0]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 64 05 b0 <unknown>

th.vlseg6w.v v8, (a0)
# CHECK-INST: th.vlseg6w.v v8, (a0)
# CHECK-ENCODING: [0x07,0x64,0x05,0xb2]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 64 05 b2 <unknown>

th.vlseg6bu.v v8, (a0), v0.t
# CHECK-INST: th.vlseg6bu.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x04,0x05,0xa0]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 04 05 a0 <unknown>

th.vlseg6bu.v v8, (a0)
# CHECK-INST: th.vlseg6bu.v v8, (a0)
# CHECK-ENCODING: [0x07,0x04,0x05,0xa2]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 04 05 a2 <unknown>

th.vlseg6hu.v v8, (a0), v0.t
# CHECK-INST: th.vlseg6hu.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x54,0x05,0xa0]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 54 05 a0 <unknown>

th.vlseg6hu.v v8, (a0)
# CHECK-INST: th.vlseg6hu.v v8, (a0)
# CHECK-ENCODING: [0x07,0x54,0x05,0xa2]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 54 05 a2 <unknown>

th.vlseg6wu.v v8, (a0), v0.t
# CHECK-INST: th.vlseg6wu.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x64,0x05,0xa0]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 64 05 a0 <unknown>

th.vlseg6wu.v v8, (a0)
# CHECK-INST: th.vlseg6wu.v v8, (a0)
# CHECK-ENCODING: [0x07,0x64,0x05,0xa2]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 64 05 a2 <unknown>

th.vlseg6e.v v8, (a0), v0.t
# CHECK-INST: th.vlseg6e.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x74,0x05,0xa0]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 74 05 a0 <unknown>

th.vlseg6e.v v8, (a0)
# CHECK-INST: th.vlseg6e.v v8, (a0)
# CHECK-ENCODING: [0x07,0x74,0x05,0xa2]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 74 05 a2 <unknown>

th.vsseg6b.v v8, (a0), v0.t
# CHECK-INST: th.vsseg6b.v v8, (a0), v0.t
# CHECK-ENCODING: [0x27,0x04,0x05,0xa0]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 04 05 a0 <unknown>

th.vsseg6b.v v8, (a0)
# CHECK-INST: th.vsseg6b.v v8, (a0)
# CHECK-ENCODING: [0x27,0x04,0x05,0xa2]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 04 05 a2 <unknown>

th.vsseg6h.v v8, (a0), v0.t
# CHECK-INST: th.vsseg6h.v v8, (a0), v0.t
# CHECK-ENCODING: [0x27,0x54,0x05,0xa0]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 54 05 a0 <unknown>

th.vsseg6h.v v8, (a0)
# CHECK-INST: th.vsseg6h.v v8, (a0)
# CHECK-ENCODING: [0x27,0x54,0x05,0xa2]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 54 05 a2 <unknown>

th.vsseg6w.v v8, (a0), v0.t
# CHECK-INST: th.vsseg6w.v v8, (a0), v0.t
# CHECK-ENCODING: [0x27,0x64,0x05,0xa0]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 64 05 a0 <unknown>

th.vsseg6w.v v8, (a0)
# CHECK-INST: th.vsseg6w.v v8, (a0)
# CHECK-ENCODING: [0x27,0x64,0x05,0xa2]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 64 05 a2 <unknown>

th.vsseg6e.v v8, (a0), v0.t
# CHECK-INST: th.vsseg6e.v v8, (a0), v0.t
# CHECK-ENCODING: [0x27,0x74,0x05,0xa0]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 74 05 a0 <unknown>

th.vsseg6e.v v8, (a0)
# CHECK-INST: th.vsseg6e.v v8, (a0)
# CHECK-ENCODING: [0x27,0x74,0x05,0xa2]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 74 05 a2 <unknown>

th.vlseg6bff.v	v8, (a0)
# CHECK-INST: th.vlseg6bff.v	v8, (a0)
# CHECK-ENCODING: [0x07,0x04,0x05,0xb3]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 04 05 b3 <unknown>

th.vlseg6bff.v	v8, (a0), v0.t
# CHECK-INST: th.vlseg6bff.v	v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x04,0x05,0xb1]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 04 05 b1 <unknown>

th.vlseg6hff.v	v8, (a0)
# CHECK-INST: th.vlseg6hff.v	v8, (a0)
# CHECK-ENCODING: [0x07,0x54,0x05,0xb3]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 54 05 b3 <unknown>

th.vlseg6hff.v	v8, (a0), v0.t
# CHECK-INST: th.vlseg6hff.v	v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x54,0x05,0xb1]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 54 05 b1 <unknown>

th.vlseg6wff.v	v8, (a0)
# CHECK-INST: th.vlseg6wff.v	v8, (a0)
# CHECK-ENCODING: [0x07,0x64,0x05,0xb3]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 64 05 b3 <unknown>

th.vlseg6wff.v	v8, (a0), v0.t
# CHECK-INST: th.vlseg6wff.v	v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x64,0x05,0xb1]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 64 05 b1 <unknown>

th.vlseg6buff.v v8, (a0)
# CHECK-INST: th.vlseg6buff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x04,0x05,0xa3]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 04 05 a3 <unknown>

th.vlseg6buff.v v8, (a0), v0.t
# CHECK-INST: th.vlseg6buff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x04,0x05,0xa1]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 04 05 a1 <unknown>

th.vlseg6huff.v v8, (a0)
# CHECK-INST: th.vlseg6huff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x54,0x05,0xa3]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 54 05 a3 <unknown>

th.vlseg6huff.v v8, (a0), v0.t
# CHECK-INST: th.vlseg6huff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x54,0x05,0xa1]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 54 05 a1 <unknown>

th.vlseg6wuff.v v8, (a0)
# CHECK-INST: th.vlseg6wuff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x64,0x05,0xa3]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 64 05 a3 <unknown>

th.vlseg6wuff.v v8, (a0), v0.t
# CHECK-INST: th.vlseg6wuff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x64,0x05,0xa1]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 64 05 a1 <unknown>

th.vlseg6eff.v	v8, (a0)
# CHECK-INST: th.vlseg6eff.v	v8, (a0)
# CHECK-ENCODING: [0x07,0x74,0x05,0xa3]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 74 05 a3 <unknown>

th.vlseg6eff.v	v8, (a0), v0.t
# CHECK-INST: th.vlseg6eff.v	v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x74,0x05,0xa1]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 74 05 a1 <unknown>

th.vlsseg6b.v v8, (a0), a1, v0.t
# CHECK-INST: th.vlsseg6b.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x04,0xb5,0xb8]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 04 b5 b8 <unknown>

th.vlsseg6b.v v8, (a0), a1
# CHECK-INST: th.vlsseg6b.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x04,0xb5,0xba]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 04 b5 ba <unknown>

th.vlsseg6h.v v8, (a0), a1, v0.t
# CHECK-INST: th.vlsseg6h.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x54,0xb5,0xb8]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 54 b5 b8 <unknown>

th.vlsseg6h.v v8, (a0), a1
# CHECK-INST: th.vlsseg6h.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x54,0xb5,0xba]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 54 b5 ba <unknown>

th.vlsseg6w.v v8, (a0), a1, v0.t
# CHECK-INST: th.vlsseg6w.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x64,0xb5,0xb8]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 64 b5 b8 <unknown>

th.vlsseg6w.v v8, (a0), a1
# CHECK-INST: th.vlsseg6w.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x64,0xb5,0xba]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 64 b5 ba <unknown>

th.vlsseg6bu.v v8, (a0), a1, v0.t
# CHECK-INST: th.vlsseg6bu.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x04,0xb5,0xa8]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 04 b5 a8 <unknown>

th.vlsseg6bu.v v8, (a0), a1
# CHECK-INST: th.vlsseg6bu.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x04,0xb5,0xaa]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 04 b5 aa <unknown>

th.vlsseg6hu.v v8, (a0), a1, v0.t
# CHECK-INST: th.vlsseg6hu.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x54,0xb5,0xa8]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 54 b5 a8 <unknown>

th.vlsseg6hu.v v8, (a0), a1
# CHECK-INST: th.vlsseg6hu.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x54,0xb5,0xaa]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 54 b5 aa <unknown>

th.vlsseg6wu.v v8, (a0), a1, v0.t
# CHECK-INST: th.vlsseg6wu.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x64,0xb5,0xa8]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 64 b5 a8 <unknown>

th.vlsseg6wu.v v8, (a0), a1
# CHECK-INST: th.vlsseg6wu.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x64,0xb5,0xaa]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 64 b5 aa <unknown>

th.vlsseg6e.v v8, (a0), a1, v0.t
# CHECK-INST: th.vlsseg6e.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x74,0xb5,0xa8]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 74 b5 a8 <unknown>

th.vlsseg6e.v v8, (a0), a1
# CHECK-INST: th.vlsseg6e.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x74,0xb5,0xaa]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 74 b5 aa <unknown>

th.vssseg6b.v	v8, (a0), a1
# CHECK-INST: th.vssseg6b.v	v8, (a0), a1
# CHECK-ENCODING: [0x27,0x04,0xb5,0xaa]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 04 b5 aa <unknown>

th.vssseg6b.v	v8, (a0), a1, v0.t
# CHECK-INST: th.vssseg6b.v	v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x04,0xb5,0xa8]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 04 b5 a8 <unknown>

th.vssseg6h.v	v8, (a0), a1
# CHECK-INST: th.vssseg6h.v	v8, (a0), a1
# CHECK-ENCODING: [0x27,0x54,0xb5,0xaa]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 54 b5 aa <unknown>

th.vssseg6h.v	v8, (a0), a1, v0.t
# CHECK-INST: th.vssseg6h.v	v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x54,0xb5,0xa8]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 54 b5 a8 <unknown>

th.vssseg6w.v	v8, (a0), a1
# CHECK-INST: th.vssseg6w.v	v8, (a0), a1
# CHECK-ENCODING: [0x27,0x64,0xb5,0xaa]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 64 b5 aa <unknown>

th.vssseg6w.v	v8, (a0), a1, v0.t
# CHECK-INST: th.vssseg6w.v	v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x64,0xb5,0xa8]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 64 b5 a8 <unknown>

th.vssseg6e.v	v8, (a0), a1
# CHECK-INST: th.vssseg6e.v	v8, (a0), a1
# CHECK-ENCODING: [0x27,0x74,0xb5,0xaa]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 74 b5 aa <unknown>

th.vssseg6e.v	v8, (a0), a1, v0.t
# CHECK-INST: th.vssseg6e.v	v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x74,0xb5,0xa8]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 74 b5 a8 <unknown>

th.vlxseg6b.v v8, (a0), v4, v0.t
# CHECK-INST: th.vlxseg6b.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x04,0x45,0xbc]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 04 45 bc <unknown>

th.vlxseg6b.v v8, (a0), v4
# CHECK-INST: th.vlxseg6b.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x04,0x45,0xbe]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 04 45 be <unknown>

th.vlxseg6h.v v8, (a0), v4, v0.t
# CHECK-INST: th.vlxseg6h.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x54,0x45,0xbc]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 54 45 bc <unknown>

th.vlxseg6h.v v8, (a0), v4
# CHECK-INST: th.vlxseg6h.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x54,0x45,0xbe]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 54 45 be <unknown>

th.vlxseg6w.v v8, (a0), v4, v0.t
# CHECK-INST: th.vlxseg6w.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x64,0x45,0xbc]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 64 45 bc <unknown>

th.vlxseg6w.v v8, (a0), v4
# CHECK-INST: th.vlxseg6w.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x64,0x45,0xbe]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 64 45 be <unknown>

th.vlxseg6bu.v v8, (a0), v4, v0.t
# CHECK-INST: th.vlxseg6bu.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x04,0x45,0xac]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 04 45 ac <unknown>

th.vlxseg6bu.v v8, (a0), v4
# CHECK-INST: th.vlxseg6bu.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x04,0x45,0xae]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 04 45 ae <unknown>

th.vlxseg6hu.v v8, (a0), v4, v0.t
# CHECK-INST: th.vlxseg6hu.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x54,0x45,0xac]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 54 45 ac <unknown>

th.vlxseg6hu.v v8, (a0), v4
# CHECK-INST: th.vlxseg6hu.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x54,0x45,0xae]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 54 45 ae <unknown>

th.vlxseg6wu.v v8, (a0), v4, v0.t
# CHECK-INST: th.vlxseg6wu.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x64,0x45,0xac]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 64 45 ac <unknown>

th.vlxseg6wu.v v8, (a0), v4
# CHECK-INST: th.vlxseg6wu.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x64,0x45,0xae]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 64 45 ae <unknown>

th.vlxseg6e.v v8, (a0), v4, v0.t
# CHECK-INST: th.vlxseg6e.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x74,0x45,0xac]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 74 45 ac <unknown>

th.vlxseg6e.v v8, (a0), v4
# CHECK-INST: th.vlxseg6e.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x74,0x45,0xae]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 74 45 ae <unknown>

th.vsxseg6b.v	v8, (a0), v4
# CHECK-INST: th.vsxseg6b.v	v8, (a0), v4
# CHECK-ENCODING: [0x27,0x04,0x45,0xae]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 04 45 ae <unknown>

th.vsxseg6b.v	v8, (a0), v4, v0.t
# CHECK-INST: th.vsxseg6b.v	v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x04,0x45,0xac]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 04 45 ac <unknown>

th.vsxseg6h.v	v8, (a0), v4
# CHECK-INST: th.vsxseg6h.v	v8, (a0), v4
# CHECK-ENCODING: [0x27,0x54,0x45,0xae]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 54 45 ae <unknown>

th.vsxseg6h.v	v8, (a0), v4, v0.t
# CHECK-INST: th.vsxseg6h.v	v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x54,0x45,0xac]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 54 45 ac <unknown>

th.vsxseg6w.v	v8, (a0), v4
# CHECK-INST: th.vsxseg6w.v	v8, (a0), v4
# CHECK-ENCODING: [0x27,0x64,0x45,0xae]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 64 45 ae <unknown>

th.vsxseg6w.v	v8, (a0), v4, v0.t
# CHECK-INST: th.vsxseg6w.v	v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x64,0x45,0xac]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 64 45 ac <unknown>

th.vsxseg6e.v	v8, (a0), v4
# CHECK-INST: th.vsxseg6e.v	v8, (a0), v4
# CHECK-ENCODING: [0x27,0x74,0x45,0xae]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 74 45 ae <unknown>

th.vsxseg6e.v	v8, (a0), v4, v0.t
# CHECK-INST: th.vsxseg6e.v	v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x74,0x45,0xac]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 74 45 ac <unknown>

th.vlseg7b.v v8, (a0), v0.t
# CHECK-INST: th.vlseg7b.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x04,0x05,0xd0]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 04 05 d0 <unknown>

th.vlseg7b.v v8, (a0)
# CHECK-INST: th.vlseg7b.v v8, (a0)
# CHECK-ENCODING: [0x07,0x04,0x05,0xd2]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 04 05 d2 <unknown>

th.vlseg7h.v v8, (a0), v0.t
# CHECK-INST: th.vlseg7h.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x54,0x05,0xd0]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 54 05 d0 <unknown>

th.vlseg7h.v v8, (a0)
# CHECK-INST: th.vlseg7h.v v8, (a0)
# CHECK-ENCODING: [0x07,0x54,0x05,0xd2]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 54 05 d2 <unknown>

th.vlseg7w.v v8, (a0), v0.t
# CHECK-INST: th.vlseg7w.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x64,0x05,0xd0]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 64 05 d0 <unknown>

th.vlseg7w.v v8, (a0)
# CHECK-INST: th.vlseg7w.v v8, (a0)
# CHECK-ENCODING: [0x07,0x64,0x05,0xd2]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 64 05 d2 <unknown>

th.vlseg7bu.v v8, (a0), v0.t
# CHECK-INST: th.vlseg7bu.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x04,0x05,0xc0]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 04 05 c0 <unknown>

th.vlseg7bu.v v8, (a0)
# CHECK-INST: th.vlseg7bu.v v8, (a0)
# CHECK-ENCODING: [0x07,0x04,0x05,0xc2]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 04 05 c2 <unknown>

th.vlseg7hu.v v8, (a0), v0.t
# CHECK-INST: th.vlseg7hu.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x54,0x05,0xc0]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 54 05 c0 <unknown>

th.vlseg7hu.v v8, (a0)
# CHECK-INST: th.vlseg7hu.v v8, (a0)
# CHECK-ENCODING: [0x07,0x54,0x05,0xc2]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 54 05 c2 <unknown>

th.vlseg7wu.v v8, (a0), v0.t
# CHECK-INST: th.vlseg7wu.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x64,0x05,0xc0]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 64 05 c0 <unknown>

th.vlseg7wu.v v8, (a0)
# CHECK-INST: th.vlseg7wu.v v8, (a0)
# CHECK-ENCODING: [0x07,0x64,0x05,0xc2]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 64 05 c2 <unknown>

th.vlseg7e.v v8, (a0), v0.t
# CHECK-INST: th.vlseg7e.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x74,0x05,0xc0]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 74 05 c0 <unknown>

th.vlseg7e.v v8, (a0)
# CHECK-INST: th.vlseg7e.v v8, (a0)
# CHECK-ENCODING: [0x07,0x74,0x05,0xc2]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 74 05 c2 <unknown>

th.vsseg7b.v v8, (a0), v0.t
# CHECK-INST: th.vsseg7b.v v8, (a0), v0.t
# CHECK-ENCODING: [0x27,0x04,0x05,0xc0]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 04 05 c0 <unknown>

th.vsseg7b.v v8, (a0)
# CHECK-INST: th.vsseg7b.v v8, (a0)
# CHECK-ENCODING: [0x27,0x04,0x05,0xc2]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 04 05 c2 <unknown>

th.vsseg7h.v v8, (a0), v0.t
# CHECK-INST: th.vsseg7h.v v8, (a0), v0.t
# CHECK-ENCODING: [0x27,0x54,0x05,0xc0]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 54 05 c0 <unknown>

th.vsseg7h.v v8, (a0)
# CHECK-INST: th.vsseg7h.v v8, (a0)
# CHECK-ENCODING: [0x27,0x54,0x05,0xc2]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 54 05 c2 <unknown>

th.vsseg7w.v v8, (a0), v0.t
# CHECK-INST: th.vsseg7w.v v8, (a0), v0.t
# CHECK-ENCODING: [0x27,0x64,0x05,0xc0]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 64 05 c0 <unknown>

th.vsseg7w.v v8, (a0)
# CHECK-INST: th.vsseg7w.v v8, (a0)
# CHECK-ENCODING: [0x27,0x64,0x05,0xc2]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 64 05 c2 <unknown>

th.vsseg7e.v v8, (a0), v0.t
# CHECK-INST: th.vsseg7e.v v8, (a0), v0.t
# CHECK-ENCODING: [0x27,0x74,0x05,0xc0]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 74 05 c0 <unknown>

th.vsseg7e.v v8, (a0)
# CHECK-INST: th.vsseg7e.v v8, (a0)
# CHECK-ENCODING: [0x27,0x74,0x05,0xc2]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 74 05 c2 <unknown>

th.vlseg7bff.v	v8, (a0)
# CHECK-INST: th.vlseg7bff.v	v8, (a0)
# CHECK-ENCODING: [0x07,0x04,0x05,0xd3]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 04 05 d3 <unknown>

th.vlseg7bff.v	v8, (a0), v0.t
# CHECK-INST: th.vlseg7bff.v	v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x04,0x05,0xd1]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 04 05 d1 <unknown>

th.vlseg7hff.v	v8, (a0)
# CHECK-INST: th.vlseg7hff.v	v8, (a0)
# CHECK-ENCODING: [0x07,0x54,0x05,0xd3]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 54 05 d3 <unknown>

th.vlseg7hff.v	v8, (a0), v0.t
# CHECK-INST: th.vlseg7hff.v	v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x54,0x05,0xd1]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 54 05 d1 <unknown>

th.vlseg7wff.v	v8, (a0)
# CHECK-INST: th.vlseg7wff.v	v8, (a0)
# CHECK-ENCODING: [0x07,0x64,0x05,0xd3]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 64 05 d3 <unknown>

th.vlseg7wff.v	v8, (a0), v0.t
# CHECK-INST: th.vlseg7wff.v	v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x64,0x05,0xd1]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 64 05 d1 <unknown>

th.vlseg7buff.v v8, (a0)
# CHECK-INST: th.vlseg7buff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x04,0x05,0xc3]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 04 05 c3 <unknown>

th.vlseg7buff.v v8, (a0), v0.t
# CHECK-INST: th.vlseg7buff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x04,0x05,0xc1]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 04 05 c1 <unknown>

th.vlseg7huff.v v8, (a0)
# CHECK-INST: th.vlseg7huff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x54,0x05,0xc3]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 54 05 c3 <unknown>

th.vlseg7huff.v v8, (a0), v0.t
# CHECK-INST: th.vlseg7huff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x54,0x05,0xc1]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 54 05 c1 <unknown>

th.vlseg7wuff.v v8, (a0)
# CHECK-INST: th.vlseg7wuff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x64,0x05,0xc3]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 64 05 c3 <unknown>

th.vlseg7wuff.v v8, (a0), v0.t
# CHECK-INST: th.vlseg7wuff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x64,0x05,0xc1]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 64 05 c1 <unknown>

th.vlseg7eff.v	v8, (a0)
# CHECK-INST: th.vlseg7eff.v	v8, (a0)
# CHECK-ENCODING: [0x07,0x74,0x05,0xc3]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 74 05 c3 <unknown>

th.vlseg7eff.v	v8, (a0), v0.t
# CHECK-INST: th.vlseg7eff.v	v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x74,0x05,0xc1]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 74 05 c1 <unknown>

th.vlsseg7b.v v8, (a0), a1, v0.t
# CHECK-INST: th.vlsseg7b.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x04,0xb5,0xd8]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 04 b5 d8 <unknown>

th.vlsseg7b.v v8, (a0), a1
# CHECK-INST: th.vlsseg7b.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x04,0xb5,0xda]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 04 b5 da <unknown>

th.vlsseg7h.v v8, (a0), a1, v0.t
# CHECK-INST: th.vlsseg7h.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x54,0xb5,0xd8]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 54 b5 d8 <unknown>

th.vlsseg7h.v v8, (a0), a1
# CHECK-INST: th.vlsseg7h.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x54,0xb5,0xda]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 54 b5 da <unknown>

th.vlsseg7w.v v8, (a0), a1, v0.t
# CHECK-INST: th.vlsseg7w.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x64,0xb5,0xd8]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 64 b5 d8 <unknown>

th.vlsseg7w.v v8, (a0), a1
# CHECK-INST: th.vlsseg7w.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x64,0xb5,0xda]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 64 b5 da <unknown>

th.vlsseg7bu.v v8, (a0), a1, v0.t
# CHECK-INST: th.vlsseg7bu.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x04,0xb5,0xc8]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 04 b5 c8 <unknown>

th.vlsseg7bu.v v8, (a0), a1
# CHECK-INST: th.vlsseg7bu.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x04,0xb5,0xca]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 04 b5 ca <unknown>

th.vlsseg7hu.v v8, (a0), a1, v0.t
# CHECK-INST: th.vlsseg7hu.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x54,0xb5,0xc8]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 54 b5 c8 <unknown>

th.vlsseg7hu.v v8, (a0), a1
# CHECK-INST: th.vlsseg7hu.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x54,0xb5,0xca]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 54 b5 ca <unknown>

th.vlsseg7wu.v v8, (a0), a1, v0.t
# CHECK-INST: th.vlsseg7wu.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x64,0xb5,0xc8]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 64 b5 c8 <unknown>

th.vlsseg7wu.v v8, (a0), a1
# CHECK-INST: th.vlsseg7wu.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x64,0xb5,0xca]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 64 b5 ca <unknown>

th.vlsseg7e.v v8, (a0), a1, v0.t
# CHECK-INST: th.vlsseg7e.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x74,0xb5,0xc8]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 74 b5 c8 <unknown>

th.vlsseg7e.v v8, (a0), a1
# CHECK-INST: th.vlsseg7e.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x74,0xb5,0xca]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 74 b5 ca <unknown>

th.vssseg7b.v	v8, (a0), a1
# CHECK-INST: th.vssseg7b.v	v8, (a0), a1
# CHECK-ENCODING: [0x27,0x04,0xb5,0xca]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 04 b5 ca <unknown>

th.vssseg7b.v	v8, (a0), a1, v0.t
# CHECK-INST: th.vssseg7b.v	v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x04,0xb5,0xc8]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 04 b5 c8 <unknown>

th.vssseg7h.v	v8, (a0), a1
# CHECK-INST: th.vssseg7h.v	v8, (a0), a1
# CHECK-ENCODING: [0x27,0x54,0xb5,0xca]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 54 b5 ca <unknown>

th.vssseg7h.v	v8, (a0), a1, v0.t
# CHECK-INST: th.vssseg7h.v	v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x54,0xb5,0xc8]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 54 b5 c8 <unknown>

th.vssseg7w.v	v8, (a0), a1
# CHECK-INST: th.vssseg7w.v	v8, (a0), a1
# CHECK-ENCODING: [0x27,0x64,0xb5,0xca]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 64 b5 ca <unknown>

th.vssseg7w.v	v8, (a0), a1, v0.t
# CHECK-INST: th.vssseg7w.v	v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x64,0xb5,0xc8]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 64 b5 c8 <unknown>

th.vssseg7e.v	v8, (a0), a1
# CHECK-INST: th.vssseg7e.v	v8, (a0), a1
# CHECK-ENCODING: [0x27,0x74,0xb5,0xca]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 74 b5 ca <unknown>

th.vssseg7e.v	v8, (a0), a1, v0.t
# CHECK-INST: th.vssseg7e.v	v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x74,0xb5,0xc8]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 74 b5 c8 <unknown>

th.vlxseg7b.v v8, (a0), v4, v0.t
# CHECK-INST: th.vlxseg7b.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x04,0x45,0xdc]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 04 45 dc <unknown>

th.vlxseg7b.v v8, (a0), v4
# CHECK-INST: th.vlxseg7b.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x04,0x45,0xde]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 04 45 de <unknown>

th.vlxseg7h.v v8, (a0), v4, v0.t
# CHECK-INST: th.vlxseg7h.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x54,0x45,0xdc]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 54 45 dc <unknown>

th.vlxseg7h.v v8, (a0), v4
# CHECK-INST: th.vlxseg7h.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x54,0x45,0xde]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 54 45 de <unknown>

th.vlxseg7w.v v8, (a0), v4, v0.t
# CHECK-INST: th.vlxseg7w.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x64,0x45,0xdc]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 64 45 dc <unknown>

th.vlxseg7w.v v8, (a0), v4
# CHECK-INST: th.vlxseg7w.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x64,0x45,0xde]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 64 45 de <unknown>

th.vlxseg7bu.v v8, (a0), v4, v0.t
# CHECK-INST: th.vlxseg7bu.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x04,0x45,0xcc]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 04 45 cc <unknown>

th.vlxseg7bu.v v8, (a0), v4
# CHECK-INST: th.vlxseg7bu.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x04,0x45,0xce]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 04 45 ce <unknown>

th.vlxseg7hu.v v8, (a0), v4, v0.t
# CHECK-INST: th.vlxseg7hu.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x54,0x45,0xcc]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 54 45 cc <unknown>

th.vlxseg7hu.v v8, (a0), v4
# CHECK-INST: th.vlxseg7hu.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x54,0x45,0xce]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 54 45 ce <unknown>

th.vlxseg7wu.v v8, (a0), v4, v0.t
# CHECK-INST: th.vlxseg7wu.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x64,0x45,0xcc]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 64 45 cc <unknown>

th.vlxseg7wu.v v8, (a0), v4
# CHECK-INST: th.vlxseg7wu.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x64,0x45,0xce]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 64 45 ce <unknown>

th.vlxseg7e.v v8, (a0), v4, v0.t
# CHECK-INST: th.vlxseg7e.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x74,0x45,0xcc]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 74 45 cc <unknown>

th.vlxseg7e.v v8, (a0), v4
# CHECK-INST: th.vlxseg7e.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x74,0x45,0xce]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 74 45 ce <unknown>

th.vsxseg7b.v	v8, (a0), v4
# CHECK-INST: th.vsxseg7b.v	v8, (a0), v4
# CHECK-ENCODING: [0x27,0x04,0x45,0xce]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 04 45 ce <unknown>

th.vsxseg7b.v	v8, (a0), v4, v0.t
# CHECK-INST: th.vsxseg7b.v	v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x04,0x45,0xcc]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 04 45 cc <unknown>

th.vsxseg7h.v	v8, (a0), v4
# CHECK-INST: th.vsxseg7h.v	v8, (a0), v4
# CHECK-ENCODING: [0x27,0x54,0x45,0xce]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 54 45 ce <unknown>

th.vsxseg7h.v	v8, (a0), v4, v0.t
# CHECK-INST: th.vsxseg7h.v	v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x54,0x45,0xcc]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 54 45 cc <unknown>

th.vsxseg7w.v	v8, (a0), v4
# CHECK-INST: th.vsxseg7w.v	v8, (a0), v4
# CHECK-ENCODING: [0x27,0x64,0x45,0xce]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 64 45 ce <unknown>

th.vsxseg7w.v	v8, (a0), v4, v0.t
# CHECK-INST: th.vsxseg7w.v	v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x64,0x45,0xcc]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 64 45 cc <unknown>

th.vsxseg7e.v	v8, (a0), v4
# CHECK-INST: th.vsxseg7e.v	v8, (a0), v4
# CHECK-ENCODING: [0x27,0x74,0x45,0xce]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 74 45 ce <unknown>

th.vsxseg7e.v	v8, (a0), v4, v0.t
# CHECK-INST: th.vsxseg7e.v	v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x74,0x45,0xcc]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 74 45 cc <unknown>

th.vlseg8b.v v8, (a0), v0.t
# CHECK-INST: th.vlseg8b.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x04,0x05,0xf0]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 04 05 f0 <unknown>

th.vlseg8b.v v8, (a0)
# CHECK-INST: th.vlseg8b.v v8, (a0)
# CHECK-ENCODING: [0x07,0x04,0x05,0xf2]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 04 05 f2 <unknown>

th.vlseg8h.v v8, (a0), v0.t
# CHECK-INST: th.vlseg8h.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x54,0x05,0xf0]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 54 05 f0 <unknown>

th.vlseg8h.v v8, (a0)
# CHECK-INST: th.vlseg8h.v v8, (a0)
# CHECK-ENCODING: [0x07,0x54,0x05,0xf2]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 54 05 f2 <unknown>

th.vlseg8w.v v8, (a0), v0.t
# CHECK-INST: th.vlseg8w.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x64,0x05,0xf0]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 64 05 f0 <unknown>

th.vlseg8w.v v8, (a0)
# CHECK-INST: th.vlseg8w.v v8, (a0)
# CHECK-ENCODING: [0x07,0x64,0x05,0xf2]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 64 05 f2 <unknown>

th.vlseg8bu.v v8, (a0), v0.t
# CHECK-INST: th.vlseg8bu.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x04,0x05,0xe0]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 04 05 e0 <unknown>

th.vlseg8bu.v v8, (a0)
# CHECK-INST: th.vlseg8bu.v v8, (a0)
# CHECK-ENCODING: [0x07,0x04,0x05,0xe2]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 04 05 e2 <unknown>

th.vlseg8hu.v v8, (a0), v0.t
# CHECK-INST: th.vlseg8hu.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x54,0x05,0xe0]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 54 05 e0 <unknown>

th.vlseg8hu.v v8, (a0)
# CHECK-INST: th.vlseg8hu.v v8, (a0)
# CHECK-ENCODING: [0x07,0x54,0x05,0xe2]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 54 05 e2 <unknown>

th.vlseg8wu.v v8, (a0), v0.t
# CHECK-INST: th.vlseg8wu.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x64,0x05,0xe0]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 64 05 e0 <unknown>

th.vlseg8wu.v v8, (a0)
# CHECK-INST: th.vlseg8wu.v v8, (a0)
# CHECK-ENCODING: [0x07,0x64,0x05,0xe2]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 64 05 e2 <unknown>

th.vlseg8e.v v8, (a0), v0.t
# CHECK-INST: th.vlseg8e.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x74,0x05,0xe0]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 74 05 e0 <unknown>

th.vlseg8e.v v8, (a0)
# CHECK-INST: th.vlseg8e.v v8, (a0)
# CHECK-ENCODING: [0x07,0x74,0x05,0xe2]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 74 05 e2 <unknown>

th.vsseg8b.v v8, (a0), v0.t
# CHECK-INST: th.vsseg8b.v v8, (a0), v0.t
# CHECK-ENCODING: [0x27,0x04,0x05,0xe0]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 04 05 e0 <unknown>

th.vsseg8b.v v8, (a0)
# CHECK-INST: th.vsseg8b.v v8, (a0)
# CHECK-ENCODING: [0x27,0x04,0x05,0xe2]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 04 05 e2 <unknown>

th.vsseg8h.v v8, (a0), v0.t
# CHECK-INST: th.vsseg8h.v v8, (a0), v0.t
# CHECK-ENCODING: [0x27,0x54,0x05,0xe0]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 54 05 e0 <unknown>

th.vsseg8h.v v8, (a0)
# CHECK-INST: th.vsseg8h.v v8, (a0)
# CHECK-ENCODING: [0x27,0x54,0x05,0xe2]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 54 05 e2 <unknown>

th.vsseg8w.v v8, (a0), v0.t
# CHECK-INST: th.vsseg8w.v v8, (a0), v0.t
# CHECK-ENCODING: [0x27,0x64,0x05,0xe0]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 64 05 e0 <unknown>

th.vsseg8w.v v8, (a0)
# CHECK-INST: th.vsseg8w.v v8, (a0)
# CHECK-ENCODING: [0x27,0x64,0x05,0xe2]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 64 05 e2 <unknown>

th.vsseg8e.v v8, (a0), v0.t
# CHECK-INST: th.vsseg8e.v v8, (a0), v0.t
# CHECK-ENCODING: [0x27,0x74,0x05,0xe0]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 74 05 e0 <unknown>

th.vsseg8e.v v8, (a0)
# CHECK-INST: th.vsseg8e.v v8, (a0)
# CHECK-ENCODING: [0x27,0x74,0x05,0xe2]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 74 05 e2 <unknown>

th.vlseg8bff.v	v8, (a0)
# CHECK-INST: th.vlseg8bff.v	v8, (a0)
# CHECK-ENCODING: [0x07,0x04,0x05,0xf3]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 04 05 f3 <unknown>

th.vlseg8bff.v	v8, (a0), v0.t
# CHECK-INST: th.vlseg8bff.v	v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x04,0x05,0xf1]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 04 05 f1 <unknown>

th.vlseg8hff.v	v8, (a0)
# CHECK-INST: th.vlseg8hff.v	v8, (a0)
# CHECK-ENCODING: [0x07,0x54,0x05,0xf3]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 54 05 f3 <unknown>

th.vlseg8hff.v	v8, (a0), v0.t
# CHECK-INST: th.vlseg8hff.v	v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x54,0x05,0xf1]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 54 05 f1 <unknown>

th.vlseg8wff.v	v8, (a0)
# CHECK-INST: th.vlseg8wff.v	v8, (a0)
# CHECK-ENCODING: [0x07,0x64,0x05,0xf3]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 64 05 f3 <unknown>

th.vlseg8wff.v	v8, (a0), v0.t
# CHECK-INST: th.vlseg8wff.v	v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x64,0x05,0xf1]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 64 05 f1 <unknown>

th.vlseg8buff.v v8, (a0)
# CHECK-INST: th.vlseg8buff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x04,0x05,0xe3]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 04 05 e3 <unknown>

th.vlseg8buff.v v8, (a0), v0.t
# CHECK-INST: th.vlseg8buff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x04,0x05,0xe1]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 04 05 e1 <unknown>

th.vlseg8huff.v v8, (a0)
# CHECK-INST: th.vlseg8huff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x54,0x05,0xe3]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 54 05 e3 <unknown>

th.vlseg8huff.v v8, (a0), v0.t
# CHECK-INST: th.vlseg8huff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x54,0x05,0xe1]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 54 05 e1 <unknown>

th.vlseg8wuff.v v8, (a0)
# CHECK-INST: th.vlseg8wuff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x64,0x05,0xe3]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 64 05 e3 <unknown>

th.vlseg8wuff.v v8, (a0), v0.t
# CHECK-INST: th.vlseg8wuff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x64,0x05,0xe1]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 64 05 e1 <unknown>

th.vlseg8eff.v	v8, (a0)
# CHECK-INST: th.vlseg8eff.v	v8, (a0)
# CHECK-ENCODING: [0x07,0x74,0x05,0xe3]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 74 05 e3 <unknown>

th.vlseg8eff.v	v8, (a0), v0.t
# CHECK-INST: th.vlseg8eff.v	v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x74,0x05,0xe1]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 74 05 e1 <unknown>

th.vlsseg8b.v v8, (a0), a1, v0.t
# CHECK-INST: th.vlsseg8b.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x04,0xb5,0xf8]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 04 b5 f8 <unknown>

th.vlsseg8b.v v8, (a0), a1
# CHECK-INST: th.vlsseg8b.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x04,0xb5,0xfa]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 04 b5 fa <unknown>

th.vlsseg8h.v v8, (a0), a1, v0.t
# CHECK-INST: th.vlsseg8h.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x54,0xb5,0xf8]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 54 b5 f8 <unknown>

th.vlsseg8h.v v8, (a0), a1
# CHECK-INST: th.vlsseg8h.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x54,0xb5,0xfa]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 54 b5 fa <unknown>

th.vlsseg8w.v v8, (a0), a1, v0.t
# CHECK-INST: th.vlsseg8w.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x64,0xb5,0xf8]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 64 b5 f8 <unknown>

th.vlsseg8w.v v8, (a0), a1
# CHECK-INST: th.vlsseg8w.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x64,0xb5,0xfa]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 64 b5 fa <unknown>

th.vlsseg8bu.v v8, (a0), a1, v0.t
# CHECK-INST: th.vlsseg8bu.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x04,0xb5,0xe8]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 04 b5 e8 <unknown>

th.vlsseg8bu.v v8, (a0), a1
# CHECK-INST: th.vlsseg8bu.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x04,0xb5,0xea]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 04 b5 ea <unknown>

th.vlsseg8hu.v v8, (a0), a1, v0.t
# CHECK-INST: th.vlsseg8hu.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x54,0xb5,0xe8]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 54 b5 e8 <unknown>

th.vlsseg8hu.v v8, (a0), a1
# CHECK-INST: th.vlsseg8hu.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x54,0xb5,0xea]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 54 b5 ea <unknown>

th.vlsseg8wu.v v8, (a0), a1, v0.t
# CHECK-INST: th.vlsseg8wu.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x64,0xb5,0xe8]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 64 b5 e8 <unknown>

th.vlsseg8wu.v v8, (a0), a1
# CHECK-INST: th.vlsseg8wu.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x64,0xb5,0xea]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 64 b5 ea <unknown>

th.vlsseg8e.v v8, (a0), a1, v0.t
# CHECK-INST: th.vlsseg8e.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x74,0xb5,0xe8]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 74 b5 e8 <unknown>

th.vlsseg8e.v v8, (a0), a1
# CHECK-INST: th.vlsseg8e.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x74,0xb5,0xea]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 74 b5 ea <unknown>

th.vssseg8b.v	v8, (a0), a1
# CHECK-INST: th.vssseg8b.v	v8, (a0), a1
# CHECK-ENCODING: [0x27,0x04,0xb5,0xea]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 04 b5 ea <unknown>

th.vssseg8b.v	v8, (a0), a1, v0.t
# CHECK-INST: th.vssseg8b.v	v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x04,0xb5,0xe8]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 04 b5 e8 <unknown>

th.vssseg8h.v	v8, (a0), a1
# CHECK-INST: th.vssseg8h.v	v8, (a0), a1
# CHECK-ENCODING: [0x27,0x54,0xb5,0xea]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 54 b5 ea <unknown>

th.vssseg8h.v	v8, (a0), a1, v0.t
# CHECK-INST: th.vssseg8h.v	v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x54,0xb5,0xe8]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 54 b5 e8 <unknown>

th.vssseg8w.v	v8, (a0), a1
# CHECK-INST: th.vssseg8w.v	v8, (a0), a1
# CHECK-ENCODING: [0x27,0x64,0xb5,0xea]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 64 b5 ea <unknown>

th.vssseg8w.v	v8, (a0), a1, v0.t
# CHECK-INST: th.vssseg8w.v	v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x64,0xb5,0xe8]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 64 b5 e8 <unknown>

th.vssseg8e.v	v8, (a0), a1
# CHECK-INST: th.vssseg8e.v	v8, (a0), a1
# CHECK-ENCODING: [0x27,0x74,0xb5,0xea]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 74 b5 ea <unknown>

th.vssseg8e.v	v8, (a0), a1, v0.t
# CHECK-INST: th.vssseg8e.v	v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x74,0xb5,0xe8]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 74 b5 e8 <unknown>

th.vlxseg8b.v v8, (a0), v4, v0.t
# CHECK-INST: th.vlxseg8b.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x04,0x45,0xfc]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 04 45 fc <unknown>

th.vlxseg8b.v v8, (a0), v4
# CHECK-INST: th.vlxseg8b.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x04,0x45,0xfe]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 04 45 fe <unknown>

th.vlxseg8h.v v8, (a0), v4, v0.t
# CHECK-INST: th.vlxseg8h.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x54,0x45,0xfc]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 54 45 fc <unknown>

th.vlxseg8h.v v8, (a0), v4
# CHECK-INST: th.vlxseg8h.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x54,0x45,0xfe]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 54 45 fe <unknown>

th.vlxseg8w.v v8, (a0), v4, v0.t
# CHECK-INST: th.vlxseg8w.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x64,0x45,0xfc]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 64 45 fc <unknown>

th.vlxseg8w.v v8, (a0), v4
# CHECK-INST: th.vlxseg8w.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x64,0x45,0xfe]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 64 45 fe <unknown>

th.vlxseg8bu.v v8, (a0), v4, v0.t
# CHECK-INST: th.vlxseg8bu.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x04,0x45,0xec]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 04 45 ec <unknown>

th.vlxseg8bu.v v8, (a0), v4
# CHECK-INST: th.vlxseg8bu.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x04,0x45,0xee]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 04 45 ee <unknown>

th.vlxseg8hu.v v8, (a0), v4, v0.t
# CHECK-INST: th.vlxseg8hu.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x54,0x45,0xec]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 54 45 ec <unknown>

th.vlxseg8hu.v v8, (a0), v4
# CHECK-INST: th.vlxseg8hu.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x54,0x45,0xee]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 54 45 ee <unknown>

th.vlxseg8wu.v v8, (a0), v4, v0.t
# CHECK-INST: th.vlxseg8wu.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x64,0x45,0xec]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 64 45 ec <unknown>

th.vlxseg8wu.v v8, (a0), v4
# CHECK-INST: th.vlxseg8wu.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x64,0x45,0xee]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 64 45 ee <unknown>

th.vlxseg8e.v v8, (a0), v4, v0.t
# CHECK-INST: th.vlxseg8e.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x74,0x45,0xec]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 74 45 ec <unknown>

th.vlxseg8e.v v8, (a0), v4
# CHECK-INST: th.vlxseg8e.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x74,0x45,0xee]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 07 74 45 ee <unknown>

th.vsxseg8b.v	v8, (a0), v4
# CHECK-INST: th.vsxseg8b.v	v8, (a0), v4
# CHECK-ENCODING: [0x27,0x04,0x45,0xee]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 04 45 ee <unknown>

th.vsxseg8b.v	v8, (a0), v4, v0.t
# CHECK-INST: th.vsxseg8b.v	v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x04,0x45,0xec]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 04 45 ec <unknown>

th.vsxseg8h.v	v8, (a0), v4
# CHECK-INST: th.vsxseg8h.v	v8, (a0), v4
# CHECK-ENCODING: [0x27,0x54,0x45,0xee]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 54 45 ee <unknown>

th.vsxseg8h.v	v8, (a0), v4, v0.t
# CHECK-INST: th.vsxseg8h.v	v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x54,0x45,0xec]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 54 45 ec <unknown>

th.vsxseg8w.v	v8, (a0), v4
# CHECK-INST: th.vsxseg8w.v	v8, (a0), v4
# CHECK-ENCODING: [0x27,0x64,0x45,0xee]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 64 45 ee <unknown>

th.vsxseg8w.v	v8, (a0), v4, v0.t
# CHECK-INST: th.vsxseg8w.v	v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x64,0x45,0xec]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 64 45 ec <unknown>

th.vsxseg8e.v	v8, (a0), v4
# CHECK-INST: th.vsxseg8e.v	v8, (a0), v4
# CHECK-ENCODING: [0x27,0x74,0x45,0xee]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 74 45 ee <unknown>

th.vsxseg8e.v	v8, (a0), v4, v0.t
# CHECK-INST: th.vsxseg8e.v	v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x74,0x45,0xec]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 27 74 45 ec <unknown>
