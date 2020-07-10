# RUN: llvm-mc -triple=riscv64 -show-encoding --mattr=+experimental-v %s \
# RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: not llvm-mc -triple=riscv64 -show-encoding %s 2>&1 \
# RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+experimental-v %s \
# RUN:        | llvm-objdump -d --mattr=+experimental-v - \
# RUN:        | FileCheck %s --check-prefix=CHECK-INST
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+experimental-v %s \
# RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

vredsum.vs v8, v4, v20, v0.t
# CHECK-INST: vredsum.vs v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x24,0x4a,0x00]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 24 4a 00 <unknown>

vredsum.vs v8, v4, v20
# CHECK-INST: vredsum.vs v8, v4, v20
# CHECK-ENCODING: [0x57,0x24,0x4a,0x02]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 24 4a 02 <unknown>

vredmaxu.vs v8, v4, v20, v0.t
# CHECK-INST: vredmaxu.vs v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x24,0x4a,0x18]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 24 4a 18 <unknown>

vredmaxu.vs v8, v4, v20
# CHECK-INST: vredmaxu.vs v8, v4, v20
# CHECK-ENCODING: [0x57,0x24,0x4a,0x1a]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 24 4a 1a <unknown>

vredmax.vs v8, v4, v20, v0.t
# CHECK-INST: vredmax.vs v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x24,0x4a,0x1c]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 24 4a 1c <unknown>

vredmax.vs v8, v4, v20
# CHECK-INST: vredmax.vs v8, v4, v20
# CHECK-ENCODING: [0x57,0x24,0x4a,0x1e]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 24 4a 1e <unknown>

vredminu.vs v8, v4, v20, v0.t
# CHECK-INST: vredminu.vs v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x24,0x4a,0x10]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 24 4a 10 <unknown>

vredminu.vs v8, v4, v20
# CHECK-INST: vredminu.vs v8, v4, v20
# CHECK-ENCODING: [0x57,0x24,0x4a,0x12]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 24 4a 12 <unknown>

vredmin.vs v8, v4, v20, v0.t
# CHECK-INST: vredmin.vs v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x24,0x4a,0x14]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 24 4a 14 <unknown>

vredmin.vs v8, v4, v20
# CHECK-INST: vredmin.vs v8, v4, v20
# CHECK-ENCODING: [0x57,0x24,0x4a,0x16]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 24 4a 16 <unknown>

vredand.vs v8, v4, v20, v0.t
# CHECK-INST: vredand.vs v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x24,0x4a,0x04]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 24 4a 04 <unknown>

vredand.vs v8, v4, v20
# CHECK-INST: vredand.vs v8, v4, v20
# CHECK-ENCODING: [0x57,0x24,0x4a,0x06]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 24 4a 06 <unknown>

vredor.vs v8, v4, v20, v0.t
# CHECK-INST: vredor.vs v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x24,0x4a,0x08]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 24 4a 08 <unknown>

vredor.vs v8, v4, v20
# CHECK-INST: vredor.vs v8, v4, v20
# CHECK-ENCODING: [0x57,0x24,0x4a,0x0a]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 24 4a 0a <unknown>

vredxor.vs v8, v4, v20, v0.t
# CHECK-INST: vredxor.vs v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x24,0x4a,0x0c]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 24 4a 0c <unknown>

vredxor.vs v8, v4, v20
# CHECK-INST: vredxor.vs v8, v4, v20
# CHECK-ENCODING: [0x57,0x24,0x4a,0x0e]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 24 4a 0e <unknown>

vwredsumu.vs v8, v4, v20, v0.t
# CHECK-INST: vwredsumu.vs v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x04,0x4a,0xc0]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 04 4a c0 <unknown>

vwredsumu.vs v8, v4, v20
# CHECK-INST: vwredsumu.vs v8, v4, v20
# CHECK-ENCODING: [0x57,0x04,0x4a,0xc2]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 04 4a c2 <unknown>

vwredsum.vs v8, v4, v20, v0.t
# CHECK-INST: vwredsum.vs v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x04,0x4a,0xc4]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 04 4a c4 <unknown>

vwredsum.vs v8, v4, v20
# CHECK-INST: vwredsum.vs v8, v4, v20
# CHECK-ENCODING: [0x57,0x04,0x4a,0xc6]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 04 4a c6 <unknown>
