# RUN: llvm-mc -triple=riscv64 -show-encoding --mattr=+experimental-v %s \
# RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: not llvm-mc -triple=riscv64 -show-encoding %s 2>&1 \
# RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+experimental-v %s \
# RUN:        | llvm-objdump -d --mattr=+experimental-v - \
# RUN:        | FileCheck %s --check-prefix=CHECK-INST
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+experimental-v %s \
# RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

vse8.v v24, (a0), v0.t
# CHECK-INST: vse8.v v24, (a0), v0.t
# CHECK-ENCODING: [0x27,0x0c,0x05,0x00]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 27 0c 05 00 <unknown>

vse8.v v24, (a0)
# CHECK-INST: vse8.v v24, (a0)
# CHECK-ENCODING: [0x27,0x0c,0x05,0x02]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 27 0c 05 02 <unknown>

vse16.v v24, (a0), v0.t
# CHECK-INST: vse16.v v24, (a0), v0.t
# CHECK-ENCODING: [0x27,0x5c,0x05,0x00]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 27 5c 05 00 <unknown>

vse16.v v24, (a0)
# CHECK-INST: vse16.v v24, (a0)
# CHECK-ENCODING: [0x27,0x5c,0x05,0x02]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 27 5c 05 02 <unknown>

vse32.v v24, (a0), v0.t
# CHECK-INST: vse32.v v24, (a0), v0.t
# CHECK-ENCODING: [0x27,0x6c,0x05,0x00]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 27 6c 05 00 <unknown>

vse32.v v24, (a0)
# CHECK-INST: vse32.v v24, (a0)
# CHECK-ENCODING: [0x27,0x6c,0x05,0x02]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 27 6c 05 02 <unknown>

vse64.v v24, (a0), v0.t
# CHECK-INST: vse64.v v24, (a0), v0.t
# CHECK-ENCODING: [0x27,0x7c,0x05,0x00]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 27 7c 05 00 <unknown>

vse64.v v24, (a0)
# CHECK-INST: vse64.v v24, (a0)
# CHECK-ENCODING: [0x27,0x7c,0x05,0x02]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 27 7c 05 02 <unknown>

vse128.v v24, (a0), v0.t
# CHECK-INST: vse128.v v24, (a0), v0.t
# CHECK-ENCODING: [0x27,0x0c,0x05,0x10]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 27 0c 05 10 <unknown>

vse128.v v24, (a0)
# CHECK-INST: vse128.v v24, (a0)
# CHECK-ENCODING: [0x27,0x0c,0x05,0x12]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 27 0c 05 12 <unknown>

vse256.v v24, (a0), v0.t
# CHECK-INST: vse256.v v24, (a0), v0.t
# CHECK-ENCODING: [0x27,0x5c,0x05,0x10]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 27 5c 05 10 <unknown>

vse256.v v24, (a0)
# CHECK-INST: vse256.v v24, (a0)
# CHECK-ENCODING: [0x27,0x5c,0x05,0x12]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 27 5c 05 12 <unknown>

vse512.v v24, (a0), v0.t
# CHECK-INST: vse512.v v24, (a0), v0.t
# CHECK-ENCODING: [0x27,0x6c,0x05,0x10]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 27 6c 05 10 <unknown>

vse512.v v24, (a0)
# CHECK-INST: vse512.v v24, (a0)
# CHECK-ENCODING: [0x27,0x6c,0x05,0x12]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 27 6c 05 12 <unknown>

vse1024.v v24, (a0), v0.t
# CHECK-INST: vse1024.v v24, (a0), v0.t
# CHECK-ENCODING: [0x27,0x7c,0x05,0x10]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 27 7c 05 10 <unknown>

vse1024.v v24, (a0)
# CHECK-INST: vse1024.v v24, (a0)
# CHECK-ENCODING: [0x27,0x7c,0x05,0x12]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 27 7c 05 12 <unknown>

vsse8.v v24, (a0), a1, v0.t
# CHECK-INST: vsse8.v v24, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x0c,0xb5,0x08]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 27 0c b5 08 <unknown>

vsse8.v v24, (a0), a1
# CHECK-INST: vsse8.v v24, (a0), a1
# CHECK-ENCODING: [0x27,0x0c,0xb5,0x0a]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 27 0c b5 0a <unknown>

vsse16.v v24, (a0), a1, v0.t
# CHECK-INST: vsse16.v v24, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x5c,0xb5,0x08]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 27 5c b5 08 <unknown>

vsse16.v v24, (a0), a1
# CHECK-INST: vsse16.v v24, (a0), a1
# CHECK-ENCODING: [0x27,0x5c,0xb5,0x0a]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 27 5c b5 0a <unknown>

vsse32.v v24, (a0), a1, v0.t
# CHECK-INST: vsse32.v v24, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x6c,0xb5,0x08]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 27 6c b5 08 <unknown>

vsse32.v v24, (a0), a1
# CHECK-INST: vsse32.v v24, (a0), a1
# CHECK-ENCODING: [0x27,0x6c,0xb5,0x0a]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 27 6c b5 0a <unknown>

vsse64.v v24, (a0), a1, v0.t
# CHECK-INST: vsse64.v v24, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x7c,0xb5,0x08]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 27 7c b5 08 <unknown>

vsse64.v v24, (a0), a1
# CHECK-INST: vsse64.v v24, (a0), a1
# CHECK-ENCODING: [0x27,0x7c,0xb5,0x0a]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 27 7c b5 0a <unknown>

vsse128.v v24, (a0), a1, v0.t
# CHECK-INST: vsse128.v v24, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x0c,0xb5,0x18]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 27 0c b5 18 <unknown>

vsse128.v v24, (a0), a1
# CHECK-INST: vsse128.v v24, (a0), a1
# CHECK-ENCODING: [0x27,0x0c,0xb5,0x1a]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 27 0c b5 1a <unknown>

vsse256.v v24, (a0), a1, v0.t
# CHECK-INST: vsse256.v v24, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x5c,0xb5,0x18]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 27 5c b5 18 <unknown>

vsse256.v v24, (a0), a1
# CHECK-INST: vsse256.v v24, (a0), a1
# CHECK-ENCODING: [0x27,0x5c,0xb5,0x1a]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 27 5c b5 1a <unknown>

vsse512.v v24, (a0), a1, v0.t
# CHECK-INST: vsse512.v v24, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x6c,0xb5,0x18]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 27 6c b5 18 <unknown>

vsse512.v v24, (a0), a1
# CHECK-INST: vsse512.v v24, (a0), a1
# CHECK-ENCODING: [0x27,0x6c,0xb5,0x1a]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 27 6c b5 1a <unknown>

vsse1024.v v24, (a0), a1, v0.t
# CHECK-INST: vsse1024.v v24, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x7c,0xb5,0x18]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 27 7c b5 18 <unknown>

vsse1024.v v24, (a0), a1
# CHECK-INST: vsse1024.v v24, (a0), a1
# CHECK-ENCODING: [0x27,0x7c,0xb5,0x1a]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 27 7c b5 1a <unknown>

vsxei8.v v24, (a0), v4, v0.t
# CHECK-INST: vsxei8.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x0c,0x45,0x0c]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 27 0c 45 0c <unknown>

vsxei8.v v24, (a0), v4
# CHECK-INST: vsxei8.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x0c,0x45,0x0e]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 27 0c 45 0e <unknown>

vsxei16.v v24, (a0), v4, v0.t
# CHECK-INST: vsxei16.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x5c,0x45,0x0c]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 27 5c 45 0c <unknown>

vsxei16.v v24, (a0), v4
# CHECK-INST: vsxei16.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x5c,0x45,0x0e]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 27 5c 45 0e <unknown>

vsxei32.v v24, (a0), v4, v0.t
# CHECK-INST: vsxei32.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x6c,0x45,0x0c]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 27 6c 45 0c <unknown>

vsxei32.v v24, (a0), v4
# CHECK-INST: vsxei32.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x6c,0x45,0x0e]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 27 6c 45 0e <unknown>

vsxei64.v v24, (a0), v4, v0.t
# CHECK-INST: vsxei64.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x7c,0x45,0x0c]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 27 7c 45 0c <unknown>

vsxei64.v v24, (a0), v4
# CHECK-INST: vsxei64.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x7c,0x45,0x0e]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 27 7c 45 0e <unknown>

vsxei128.v v24, (a0), v4, v0.t
# CHECK-INST: vsxei128.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x0c,0x45,0x1c]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 27 0c 45 1c <unknown>

vsxei128.v v24, (a0), v4
# CHECK-INST: vsxei128.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x0c,0x45,0x1e]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 27 0c 45 1e <unknown>

vsxei256.v v24, (a0), v4, v0.t
# CHECK-INST: vsxei256.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x5c,0x45,0x1c]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 27 5c 45 1c <unknown>

vsxei256.v v24, (a0), v4
# CHECK-INST: vsxei256.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x5c,0x45,0x1e]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 27 5c 45 1e <unknown>

vsxei512.v v24, (a0), v4, v0.t
# CHECK-INST: vsxei512.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x6c,0x45,0x1c]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 27 6c 45 1c <unknown>

vsxei512.v v24, (a0), v4
# CHECK-INST: vsxei512.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x6c,0x45,0x1e]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 27 6c 45 1e <unknown>

vsxei1024.v v24, (a0), v4, v0.t
# CHECK-INST: vsxei1024.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x7c,0x45,0x1c]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 27 7c 45 1c <unknown>

vsxei1024.v v24, (a0), v4
# CHECK-INST: vsxei1024.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x7c,0x45,0x1e]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 27 7c 45 1e <unknown>

vs1r.v v24, (a0)
# CHECK-INST: vs1r.v v24, (a0)
# CHECK-ENCODING: [0x27,0x0c,0x85,0x02]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 27 0c 85 02 <unknown>
