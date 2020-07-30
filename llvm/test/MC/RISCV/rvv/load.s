# RUN: llvm-mc -triple=riscv64 -show-encoding --mattr=+experimental-v %s \
# RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: not llvm-mc -triple=riscv64 -show-encoding %s 2>&1 \
# RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+experimental-v %s \
# RUN:        | llvm-objdump -d --mattr=+experimental-v - \
# RUN:        | FileCheck %s --check-prefix=CHECK-INST
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+experimental-v %s \
# RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

vlb.v v8, (a0), v0.t
# CHECK-INST: vlb.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x04,0x05,0x10]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 07 04 05 10 <unknown>

vlb.v v8, (a0)
# CHECK-INST: vlb.v v8, (a0)
# CHECK-ENCODING: [0x07,0x04,0x05,0x12]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 07 04 05 12 <unknown>

vlh.v v8, (a0), v0.t
# CHECK-INST: vlh.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x54,0x05,0x10]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 07 54 05 10 <unknown>

vlh.v v8, (a0)
# CHECK-INST: vlh.v v8, (a0)
# CHECK-ENCODING: [0x07,0x54,0x05,0x12]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 07 54 05 12 <unknown>

vlw.v v8, (a0), v0.t
# CHECK-INST: vlw.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x64,0x05,0x10]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 07 64 05 10 <unknown>

vlw.v v8, (a0)
# CHECK-INST: vlw.v v8, (a0)
# CHECK-ENCODING: [0x07,0x64,0x05,0x12]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 07 64 05 12 <unknown>

vlbu.v v8, (a0), v0.t
# CHECK-INST: vlbu.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x04,0x05,0x00]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 07 04 05 00 <unknown>

vlbu.v v8, (a0)
# CHECK-INST: vlbu.v v8, (a0)
# CHECK-ENCODING: [0x07,0x04,0x05,0x02]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 07 04 05 02 <unknown>

vlhu.v v8, (a0), v0.t
# CHECK-INST: vlhu.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x54,0x05,0x00]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 07 54 05 00 <unknown>

vlhu.v v8, (a0)
# CHECK-INST: vlhu.v v8, (a0)
# CHECK-ENCODING: [0x07,0x54,0x05,0x02]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 07 54 05 02 <unknown>

vlwu.v v8, (a0), v0.t
# CHECK-INST: vlwu.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x64,0x05,0x00]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 07 64 05 00 <unknown>

vlwu.v v8, (a0)
# CHECK-INST: vlwu.v v8, (a0)
# CHECK-ENCODING: [0x07,0x64,0x05,0x02]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 07 64 05 02 <unknown>

vlbff.v v8, (a0), v0.t
# CHECK-INST: vlbff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x04,0x05,0x11]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 07 04 05 11 <unknown>

vlbff.v v8, (a0)
# CHECK-INST: vlbff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x04,0x05,0x13]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 07 04 05 13 <unknown>

vlhff.v v8, (a0), v0.t
# CHECK-INST: vlhff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x54,0x05,0x11]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 07 54 05 11 <unknown>

vlhff.v v8, (a0)
# CHECK-INST: vlhff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x54,0x05,0x13]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 07 54 05 13 <unknown>

vlwff.v v8, (a0), v0.t
# CHECK-INST: vlwff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x64,0x05,0x11]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 07 64 05 11 <unknown>

vlwff.v v8, (a0)
# CHECK-INST: vlwff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x64,0x05,0x13]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 07 64 05 13 <unknown>

vlbuff.v v8, (a0), v0.t
# CHECK-INST: vlbuff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x04,0x05,0x01]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 07 04 05 01 <unknown>

vlbuff.v v8, (a0)
# CHECK-INST: vlbuff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x04,0x05,0x03]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 07 04 05 03 <unknown>

vlhuff.v v8, (a0), v0.t
# CHECK-INST: vlhuff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x54,0x05,0x01]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 07 54 05 01 <unknown>

vlhuff.v v8, (a0)
# CHECK-INST: vlhuff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x54,0x05,0x03]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 07 54 05 03 <unknown>

vlwuff.v v8, (a0), v0.t
# CHECK-INST: vlwuff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x64,0x05,0x01]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 07 64 05 01 <unknown>

vlwuff.v v8, (a0)
# CHECK-INST: vlwuff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x64,0x05,0x03]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 07 64 05 03 <unknown>

vleff.v v8, (a0), v0.t
# CHECK-INST: vleff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x74,0x05,0x01]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 07 74 05 01 <unknown>

vleff.v v8, (a0)
# CHECK-INST: vleff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x74,0x05,0x03]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 07 74 05 03 <unknown>

vlsb.v v8, (a0), a1, v0.t
# CHECK-INST: vlsb.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x04,0xb5,0x18]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 07 04 b5 18 <unknown>

vlsb.v v8, (a0), a1
# CHECK-INST: vlsb.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x04,0xb5,0x1a]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 07 04 b5 1a <unknown>

vlsh.v v8, (a0), a1, v0.t
# CHECK-INST: vlsh.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x54,0xb5,0x18]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 07 54 b5 18 <unknown>

vlsh.v v8, (a0), a1
# CHECK-INST: vlsh.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x54,0xb5,0x1a]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 07 54 b5 1a <unknown>

vlsw.v v8, (a0), a1, v0.t
# CHECK-INST: vlsw.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x64,0xb5,0x18]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 07 64 b5 18 <unknown>

vlsw.v v8, (a0), a1
# CHECK-INST: vlsw.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x64,0xb5,0x1a]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 07 64 b5 1a <unknown>

vlsbu.v v8, (a0), a1, v0.t
# CHECK-INST: vlsbu.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x04,0xb5,0x08]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 07 04 b5 08 <unknown>

vlsbu.v v8, (a0), a1
# CHECK-INST: vlsbu.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x04,0xb5,0x0a]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 07 04 b5 0a <unknown>

vlshu.v v8, (a0), a1, v0.t
# CHECK-INST: vlshu.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x54,0xb5,0x08]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 07 54 b5 08 <unknown>

vlshu.v v8, (a0), a1
# CHECK-INST: vlshu.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x54,0xb5,0x0a]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 07 54 b5 0a <unknown>

vlswu.v v8, (a0), a1, v0.t
# CHECK-INST: vlswu.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x64,0xb5,0x08]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 07 64 b5 08 <unknown>

vlswu.v v8, (a0), a1
# CHECK-INST: vlswu.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x64,0xb5,0x0a]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 07 64 b5 0a <unknown>

vlse.v v8, (a0), a1, v0.t
# CHECK-INST: vlse.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x74,0xb5,0x08]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 07 74 b5 08 <unknown>

vlse.v v8, (a0), a1
# CHECK-INST: vlse.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x74,0xb5,0x0a]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 07 74 b5 0a <unknown>

vlxb.v v8, (a0), v4, v0.t
# CHECK-INST: vlxb.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x04,0x45,0x1c]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 07 04 45 1c <unknown>

vlxb.v v8, (a0), v4
# CHECK-INST: vlxb.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x04,0x45,0x1e]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 07 04 45 1e <unknown>

vlxh.v v8, (a0), v4, v0.t
# CHECK-INST: vlxh.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x54,0x45,0x1c]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 07 54 45 1c <unknown>

vlxh.v v8, (a0), v4
# CHECK-INST: vlxh.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x54,0x45,0x1e]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 07 54 45 1e <unknown>

vlxw.v v8, (a0), v4, v0.t
# CHECK-INST: vlxw.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x64,0x45,0x1c]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 07 64 45 1c <unknown>

vlxw.v v8, (a0), v4
# CHECK-INST: vlxw.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x64,0x45,0x1e]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 07 64 45 1e <unknown>

vlxbu.v v8, (a0), v4, v0.t
# CHECK-INST: vlxbu.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x04,0x45,0x0c]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 07 04 45 0c <unknown>

vlxbu.v v8, (a0), v4
# CHECK-INST: vlxbu.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x04,0x45,0x0e]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 07 04 45 0e <unknown>

vlxhu.v v8, (a0), v4, v0.t
# CHECK-INST: vlxhu.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x54,0x45,0x0c]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 07 54 45 0c <unknown>

vlxhu.v v8, (a0), v4
# CHECK-INST: vlxhu.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x54,0x45,0x0e]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 07 54 45 0e <unknown>

vlxwu.v v8, (a0), v4, v0.t
# CHECK-INST: vlxwu.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x64,0x45,0x0c]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 07 64 45 0c <unknown>

vlxwu.v v8, (a0), v4
# CHECK-INST: vlxwu.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x64,0x45,0x0e]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 07 64 45 0e <unknown>

vlxe.v v8, (a0), v4, v0.t
# CHECK-INST: vlxe.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x74,0x45,0x0c]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 07 74 45 0c <unknown>

vlxe.v v8, (a0), v4
# CHECK-INST: vlxe.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x74,0x45,0x0e]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 07 74 45 0e <unknown>

vl1r.v v8, (a0)
# CHECK-INST: vl1r.v v8, (a0)
# CHECK-ENCODING: [0x07,0x74,0x85,0x02]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 07 74 85 02 <unknown>
