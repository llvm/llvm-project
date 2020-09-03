# RUN: llvm-mc -triple=riscv64 -show-encoding --mattr=+experimental-v %s \
# RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: not llvm-mc -triple=riscv64 -show-encoding %s 2>&1 \
# RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+experimental-v %s \
# RUN:        | llvm-objdump -d --mattr=+experimental-v - \
# RUN:        | FileCheck %s --check-prefix=CHECK-INST
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+experimental-v %s \
# RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

vsb.v v24, (a0), v0.t
# CHECK-INST: vsb.v v24, (a0), v0.t
# CHECK-ENCODING: [0x27,0x0c,0x05,0x00]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 27 0c 05 00 <unknown>

vsb.v v24, (a0)
# CHECK-INST: vsb.v v24, (a0)
# CHECK-ENCODING: [0x27,0x0c,0x05,0x02]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 27 0c 05 02 <unknown>

vsh.v v24, (a0), v0.t
# CHECK-INST: vsh.v v24, (a0), v0.t
# CHECK-ENCODING: [0x27,0x5c,0x05,0x00]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 27 5c 05 00 <unknown>

vsh.v v24, (a0)
# CHECK-INST: vsh.v v24, (a0)
# CHECK-ENCODING: [0x27,0x5c,0x05,0x02]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 27 5c 05 02 <unknown>

vsw.v v24, (a0), v0.t
# CHECK-INST: vsw.v v24, (a0), v0.t
# CHECK-ENCODING: [0x27,0x6c,0x05,0x00]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 27 6c 05 00 <unknown>

vsw.v v24, (a0)
# CHECK-INST: vsw.v v24, (a0)
# CHECK-ENCODING: [0x27,0x6c,0x05,0x02]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 27 6c 05 02 <unknown>

vse.v v24, (a0), v0.t
# CHECK-INST: vse.v v24, (a0), v0.t
# CHECK-ENCODING: [0x27,0x7c,0x05,0x00]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 27 7c 05 00 <unknown>

vse.v v24, (a0)
# CHECK-INST: vse.v v24, (a0)
# CHECK-ENCODING: [0x27,0x7c,0x05,0x02]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 27 7c 05 02 <unknown>

vssb.v v24, (a0), a1, v0.t
# CHECK-INST: vssb.v v24, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x0c,0xb5,0x08]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 27 0c b5 08 <unknown>

vssb.v v24, (a0), a1
# CHECK-INST: vssb.v v24, (a0), a1
# CHECK-ENCODING: [0x27,0x0c,0xb5,0x0a]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 27 0c b5 0a <unknown>

vssh.v v24, (a0), a1, v0.t
# CHECK-INST: vssh.v v24, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x5c,0xb5,0x08]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 27 5c b5 08 <unknown>

vssh.v v24, (a0), a1
# CHECK-INST: vssh.v v24, (a0), a1
# CHECK-ENCODING: [0x27,0x5c,0xb5,0x0a]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 27 5c b5 0a <unknown>

vssw.v v24, (a0), a1, v0.t
# CHECK-INST: vssw.v v24, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x6c,0xb5,0x08]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 27 6c b5 08 <unknown>

vssw.v v24, (a0), a1
# CHECK-INST: vssw.v v24, (a0), a1
# CHECK-ENCODING: [0x27,0x6c,0xb5,0x0a]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 27 6c b5 0a <unknown>

vsse.v v24, (a0), a1, v0.t
# CHECK-INST: vsse.v v24, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x7c,0xb5,0x08]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 27 7c b5 08 <unknown>

vsse.v v24, (a0), a1
# CHECK-INST: vsse.v v24, (a0), a1
# CHECK-ENCODING: [0x27,0x7c,0xb5,0x0a]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 27 7c b5 0a <unknown>

vsxb.v v24, (a0), v4, v0.t
# CHECK-INST: vsxb.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x0c,0x45,0x0c]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 27 0c 45 0c <unknown>

vsxb.v v24, (a0), v4
# CHECK-INST: vsxb.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x0c,0x45,0x0e]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 27 0c 45 0e <unknown>

vsxh.v v24, (a0), v4, v0.t
# CHECK-INST: vsxh.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x5c,0x45,0x0c]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 27 5c 45 0c <unknown>

vsxh.v v24, (a0), v4
# CHECK-INST: vsxh.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x5c,0x45,0x0e]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 27 5c 45 0e <unknown>

vsxw.v v24, (a0), v4, v0.t
# CHECK-INST: vsxw.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x6c,0x45,0x0c]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 27 6c 45 0c <unknown>

vsxw.v v24, (a0), v4
# CHECK-INST: vsxw.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x6c,0x45,0x0e]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 27 6c 45 0e <unknown>

vsxe.v v24, (a0), v4, v0.t
# CHECK-INST: vsxe.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x7c,0x45,0x0c]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 27 7c 45 0c <unknown>

vsxe.v v24, (a0), v4
# CHECK-INST: vsxe.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x7c,0x45,0x0e]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 27 7c 45 0e <unknown>

vsuxb.v v24, (a0), v4, v0.t
# CHECK-INST: vsuxb.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x0c,0x45,0x1c]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 27 0c 45 1c <unknown>

vsuxb.v v24, (a0), v4
# CHECK-INST: vsuxb.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x0c,0x45,0x1e]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 27 0c 45 1e <unknown>

vsuxh.v v24, (a0), v4, v0.t
# CHECK-INST: vsuxh.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x5c,0x45,0x1c]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 27 5c 45 1c <unknown>

vsuxh.v v24, (a0), v4
# CHECK-INST: vsuxh.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x5c,0x45,0x1e]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 27 5c 45 1e <unknown>

vsuxw.v v24, (a0), v4, v0.t
# CHECK-INST: vsuxw.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x6c,0x45,0x1c]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 27 6c 45 1c <unknown>

vsuxw.v v24, (a0), v4
# CHECK-INST: vsuxw.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x6c,0x45,0x1e]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 27 6c 45 1e <unknown>

vsuxe.v v24, (a0), v4, v0.t
# CHECK-INST: vsuxe.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x7c,0x45,0x1c]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 27 7c 45 1c <unknown>

vsuxe.v v24, (a0), v4
# CHECK-INST: vsuxe.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x7c,0x45,0x1e]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 27 7c 45 1e <unknown>

vs1r.v v24, (a0)
# CHECK-INST: vs1r.v v24, (a0)
# CHECK-ENCODING: [0x27,0x7c,0x85,0x02]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 27 7c 85 02 <unknown>
