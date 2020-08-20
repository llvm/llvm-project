# RUN: llvm-mc -triple=riscv64 -show-encoding --mattr=+experimental-v %s \
# RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: not llvm-mc -triple=riscv64 -show-encoding %s 2>&1 \
# RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+experimental-v %s \
# RUN:        | llvm-objdump -d --mattr=+experimental-v - \
# RUN:        | FileCheck %s --check-prefix=CHECK-INST
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+experimental-v %s \
# RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

vle8.v v8, (a0), v0.t
# CHECK-INST: vle8.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x04,0x05,0x00]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 07 04 05 00 <unknown>

vle8.v v8, (a0)
# CHECK-INST: vle8.v v8, (a0)
# CHECK-ENCODING: [0x07,0x04,0x05,0x02]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 07 04 05 02 <unknown>

vle16.v v8, (a0), v0.t
# CHECK-INST: vle16.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x54,0x05,0x00]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 07 54 05 00 <unknown>

vle16.v v8, (a0)
# CHECK-INST: vle16.v v8, (a0)
# CHECK-ENCODING: [0x07,0x54,0x05,0x02]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 07 54 05 02 <unknown>

vle32.v v8, (a0), v0.t
# CHECK-INST: vle32.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x64,0x05,0x00]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 07 64 05 00 <unknown>

vle32.v v8, (a0)
# CHECK-INST: vle32.v v8, (a0)
# CHECK-ENCODING: [0x07,0x64,0x05,0x02]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 07 64 05 02 <unknown>

vle64.v v8, (a0), v0.t
# CHECK-INST: vle64.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x74,0x05,0x00]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 07 74 05 00 <unknown>

vle64.v v8, (a0)
# CHECK-INST: vle64.v v8, (a0)
# CHECK-ENCODING: [0x07,0x74,0x05,0x02]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 07 74 05 02 <unknown>

vle128.v v8, (a0), v0.t
# CHECK-INST: vle128.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x04,0x05,0x10]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 07 04 05 10 <unknown>

vle128.v v8, (a0)
# CHECK-INST: vle128.v v8, (a0)
# CHECK-ENCODING: [0x07,0x04,0x05,0x12]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 07 04 05 12 <unknown>

vle256.v v8, (a0), v0.t
# CHECK-INST: vle256.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x54,0x05,0x10]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 07 54 05 10 <unknown>

vle256.v v8, (a0)
# CHECK-INST: vle256.v v8, (a0)
# CHECK-ENCODING: [0x07,0x54,0x05,0x12]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 07 54 05 12 <unknown>

vle512.v v8, (a0), v0.t
# CHECK-INST: vle512.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x64,0x05,0x10]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 07 64 05 10 <unknown>

vle512.v v8, (a0)
# CHECK-INST: vle512.v v8, (a0)
# CHECK-ENCODING: [0x07,0x64,0x05,0x12]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 07 64 05 12 <unknown>

vle1024.v v8, (a0), v0.t
# CHECK-INST: vle1024.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x74,0x05,0x10]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 07 74 05 10 <unknown>

vle1024.v v8, (a0)
# CHECK-INST: vle1024.v v8, (a0)
# CHECK-ENCODING: [0x07,0x74,0x05,0x12]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 07 74 05 12 <unknown>

vle8ff.v v8, (a0), v0.t
# CHECK-INST: vle8ff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x04,0x05,0x01]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 07 04 05 01 <unknown>

vle8ff.v v8, (a0)
# CHECK-INST: vle8ff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x04,0x05,0x03]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 07 04 05 03 <unknown>

vle16ff.v v8, (a0), v0.t
# CHECK-INST: vle16ff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x54,0x05,0x01]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 07 54 05 01 <unknown>

vle16ff.v v8, (a0)
# CHECK-INST: vle16ff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x54,0x05,0x03]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 07 54 05 03 <unknown>

vle32ff.v v8, (a0), v0.t
# CHECK-INST: vle32ff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x64,0x05,0x01]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 07 64 05 01 <unknown>

vle32ff.v v8, (a0)
# CHECK-INST: vle32ff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x64,0x05,0x03]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 07 64 05 03 <unknown>

vle64ff.v v8, (a0), v0.t
# CHECK-INST: vle64ff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x74,0x05,0x01]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 07 74 05 01 <unknown>

vle64ff.v v8, (a0)
# CHECK-INST: vle64ff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x74,0x05,0x03]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 07 74 05 03 <unknown>

vle128ff.v v8, (a0), v0.t
# CHECK-INST: vle128ff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x04,0x05,0x11]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 07 04 05 11 <unknown>

vle128ff.v v8, (a0)
# CHECK-INST: vle128ff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x04,0x05,0x13]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 07 04 05 13 <unknown>

vle256ff.v v8, (a0), v0.t
# CHECK-INST: vle256ff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x54,0x05,0x11]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 07 54 05 11 <unknown>

vle256ff.v v8, (a0)
# CHECK-INST: vle256ff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x54,0x05,0x13]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 07 54 05 13 <unknown>

vle512ff.v v8, (a0), v0.t
# CHECK-INST: vle512ff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x64,0x05,0x11]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 07 64 05 11 <unknown>

vle512ff.v v8, (a0)
# CHECK-INST: vle512ff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x64,0x05,0x13]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 07 64 05 13 <unknown>

vle1024ff.v v8, (a0), v0.t
# CHECK-INST: vle1024ff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x74,0x05,0x11]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 07 74 05 11 <unknown>

vle1024ff.v v8, (a0)
# CHECK-INST: vle1024ff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x74,0x05,0x13]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 07 74 05 13 <unknown>

vlse8.v v8, (a0), a1, v0.t
# CHECK-INST: vlse8.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x04,0xb5,0x08]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 07 04 b5 08 <unknown>

vlse8.v v8, (a0), a1
# CHECK-INST: vlse8.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x04,0xb5,0x0a]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 07 04 b5 0a <unknown>

vlse16.v v8, (a0), a1, v0.t
# CHECK-INST: vlse16.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x54,0xb5,0x08]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 07 54 b5 08 <unknown>

vlse16.v v8, (a0), a1
# CHECK-INST: vlse16.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x54,0xb5,0x0a]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 07 54 b5 0a <unknown>

vlse32.v v8, (a0), a1, v0.t
# CHECK-INST: vlse32.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x64,0xb5,0x08]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 07 64 b5 08 <unknown>

vlse32.v v8, (a0), a1
# CHECK-INST: vlse32.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x64,0xb5,0x0a]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 07 64 b5 0a <unknown>

vlse64.v v8, (a0), a1, v0.t
# CHECK-INST: vlse64.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x74,0xb5,0x08]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 07 74 b5 08 <unknown>

vlse64.v v8, (a0), a1
# CHECK-INST: vlse64.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x74,0xb5,0x0a]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 07 74 b5 0a <unknown>

vlse128.v v8, (a0), a1, v0.t
# CHECK-INST: vlse128.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x04,0xb5,0x18]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 07 04 b5 18 <unknown>

vlse128.v v8, (a0), a1
# CHECK-INST: vlse128.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x04,0xb5,0x1a]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 07 04 b5 1a <unknown>

vlse256.v v8, (a0), a1, v0.t
# CHECK-INST: vlse256.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x54,0xb5,0x18]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 07 54 b5 18 <unknown>

vlse256.v v8, (a0), a1
# CHECK-INST: vlse256.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x54,0xb5,0x1a]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 07 54 b5 1a <unknown>

vlse512.v v8, (a0), a1, v0.t
# CHECK-INST: vlse512.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x64,0xb5,0x18]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 07 64 b5 18 <unknown>

vlse512.v v8, (a0), a1
# CHECK-INST: vlse512.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x64,0xb5,0x1a]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 07 64 b5 1a <unknown>

vlse1024.v v8, (a0), a1, v0.t
# CHECK-INST: vlse1024.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x74,0xb5,0x18]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 07 74 b5 18 <unknown>

vlse1024.v v8, (a0), a1
# CHECK-INST: vlse1024.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x74,0xb5,0x1a]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 07 74 b5 1a <unknown>

vlxei8.v v8, (a0), v4, v0.t
# CHECK-INST: vlxei8.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x04,0x45,0x0c]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 07 04 45 0c <unknown>

vlxei8.v v8, (a0), v4
# CHECK-INST: vlxei8.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x04,0x45,0x0e]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 07 04 45 0e <unknown>

vlxei16.v v8, (a0), v4, v0.t
# CHECK-INST: vlxei16.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x54,0x45,0x0c]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 07 54 45 0c <unknown>

vlxei16.v v8, (a0), v4
# CHECK-INST: vlxei16.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x54,0x45,0x0e]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 07 54 45 0e <unknown>

vlxei32.v v8, (a0), v4, v0.t
# CHECK-INST: vlxei32.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x64,0x45,0x0c]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 07 64 45 0c <unknown>

vlxei32.v v8, (a0), v4
# CHECK-INST: vlxei32.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x64,0x45,0x0e]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 07 64 45 0e <unknown>

vlxei64.v v8, (a0), v4, v0.t
# CHECK-INST: vlxei64.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x74,0x45,0x0c]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 07 74 45 0c <unknown>

vlxei64.v v8, (a0), v4
# CHECK-INST: vlxei64.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x74,0x45,0x0e]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 07 74 45 0e <unknown>

vlxei128.v v8, (a0), v4, v0.t
# CHECK-INST: vlxei128.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x04,0x45,0x1c]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 07 04 45 1c <unknown>

vlxei128.v v8, (a0), v4
# CHECK-INST: vlxei128.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x04,0x45,0x1e]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 07 04 45 1e <unknown>

vlxei256.v v8, (a0), v4, v0.t
# CHECK-INST: vlxei256.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x54,0x45,0x1c]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 07 54 45 1c <unknown>

vlxei256.v v8, (a0), v4
# CHECK-INST: vlxei256.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x54,0x45,0x1e]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 07 54 45 1e <unknown>

vlxei512.v v8, (a0), v4, v0.t
# CHECK-INST: vlxei512.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x64,0x45,0x1c]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 07 64 45 1c <unknown>

vlxei512.v v8, (a0), v4
# CHECK-INST: vlxei512.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x64,0x45,0x1e]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 07 64 45 1e <unknown>

vlxei1024.v v8, (a0), v4, v0.t
# CHECK-INST: vlxei1024.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x74,0x45,0x1c]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 07 74 45 1c <unknown>

vlxei1024.v v8, (a0), v4
# CHECK-INST: vlxei1024.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x74,0x45,0x1e]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 07 74 45 1e <unknown>

vl1r.v v8, (a0)
# CHECK-INST: vl1r.v v8, (a0)
# CHECK-ENCODING: [0x07,0x04,0x85,0x02]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 07 04 85 02 <unknown>
