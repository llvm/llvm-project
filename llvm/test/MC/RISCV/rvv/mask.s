# RUN: llvm-mc -triple=riscv64 -show-encoding --mattr=+experimental-v %s \
# RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: not llvm-mc -triple=riscv64 -show-encoding %s 2>&1 \
# RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+experimental-v %s \
# RUN:        | llvm-objdump -d --mattr=+experimental-v - \
# RUN:        | FileCheck %s --check-prefix=CHECK-INST
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+experimental-v %s \
# RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

vmand.mm v8, v4, v20
# CHECK-INST: vmand.mm v8, v4, v20
# CHECK-ENCODING: [0x57,0x24,0x4a,0x66]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 24 4a 66 <unknown>

vmnand.mm v8, v4, v20
# CHECK-INST: vmnand.mm v8, v4, v20
# CHECK-ENCODING: [0x57,0x24,0x4a,0x76]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 24 4a 76 <unknown>

vmandnot.mm v8, v4, v20
# CHECK-INST: vmandnot.mm v8, v4, v20
# CHECK-ENCODING: [0x57,0x24,0x4a,0x62]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 24 4a 62 <unknown>

vmxor.mm v8, v4, v20
# CHECK-INST: vmxor.mm v8, v4, v20
# CHECK-ENCODING: [0x57,0x24,0x4a,0x6e]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 24 4a 6e <unknown>

vmor.mm v8, v4, v20
# CHECK-INST: vmor.mm v8, v4, v20
# CHECK-ENCODING: [0x57,0x24,0x4a,0x6a]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 24 4a 6a <unknown>

vmnor.mm v8, v4, v20
# CHECK-INST: vmnor.mm v8, v4, v20
# CHECK-ENCODING: [0x57,0x24,0x4a,0x7a]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 24 4a 7a <unknown>

vmornot.mm v8, v4, v20
# CHECK-INST: vmornot.mm v8, v4, v20
# CHECK-ENCODING: [0x57,0x24,0x4a,0x72]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 24 4a 72 <unknown>

vmxnor.mm v8, v4, v20
# CHECK-INST: vmxnor.mm v8, v4, v20
# CHECK-ENCODING: [0x57,0x24,0x4a,0x7e]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 24 4a 7e <unknown>

vpopc.m a2, v4, v0.t
# CHECK-INST: vpopc.m a2, v4, v0.t
# CHECK-ENCODING: [0x57,0x26,0x48,0x40]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 26 48 40 <unknown>

vpopc.m a2, v4
# CHECK-INST: vpopc.m a2, v4
# CHECK-ENCODING: [0x57,0x26,0x48,0x42]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 26 48 42 <unknown>

vfirst.m a2, v4, v0.t
# CHECK-INST: vfirst.m a2, v4, v0.t
# CHECK-ENCODING: [0x57,0xa6,0x48,0x40]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 a6 48 40 <unknown>

vfirst.m a2, v4
# CHECK-INST: vfirst.m a2, v4
# CHECK-ENCODING: [0x57,0xa6,0x48,0x42]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 a6 48 42 <unknown>

vmsbf.m v8, v4, v0.t
# CHECK-INST: vmsbf.m v8, v4, v0.t
# CHECK-ENCODING: [0x57,0xa4,0x40,0x50]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 a4 40 50 <unknown>

vmsbf.m v8, v4
# CHECK-INST: vmsbf.m v8, v4
# CHECK-ENCODING: [0x57,0xa4,0x40,0x52]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 a4 40 52 <unknown>

vmsif.m v8, v4, v0.t
# CHECK-INST: vmsif.m v8, v4, v0.t
# CHECK-ENCODING: [0x57,0xa4,0x41,0x50]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 a4 41 50 <unknown>

vmsif.m v8, v4
# CHECK-INST: vmsif.m v8, v4
# CHECK-ENCODING: [0x57,0xa4,0x41,0x52]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 a4 41 52 <unknown>

vmsof.m v8, v4, v0.t
# CHECK-INST: vmsof.m v8, v4, v0.t
# CHECK-ENCODING: [0x57,0x24,0x41,0x50]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 24 41 50 <unknown>

vmsof.m v8, v4
# CHECK-INST: vmsof.m v8, v4
# CHECK-ENCODING: [0x57,0x24,0x41,0x52]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 24 41 52 <unknown>

viota.m v8, v4, v0.t
# CHECK-INST: viota.m v8, v4, v0.t
# CHECK-ENCODING: [0x57,0x24,0x48,0x50]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 24 48 50 <unknown>

viota.m v8, v4
# CHECK-INST: viota.m v8, v4
# CHECK-ENCODING: [0x57,0x24,0x48,0x52]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 24 48 52 <unknown>

vid.v v8, v0.t
# CHECK-INST: vid.v v8, v0.t
# CHECK-ENCODING: [0x57,0xa4,0x08,0x50]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 a4 08 50 <unknown>

vid.v v8
# CHECK-INST: vid.v v8
# CHECK-ENCODING: [0x57,0xa4,0x08,0x52]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 a4 08 52 <unknown>

vmcpy.m v8, v4
# CHECK-INST: vmcpy.m v8, v4
# CHECK-ENCODING: [0x57,0x24,0x42,0x66]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 24 42 66 <unknown>

vmclr.m v8
# CHECK-INST: vmclr.m v8
# CHECK-ENCODING: [0x57,0x24,0x84,0x6e]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 24 84 6e <unknown>

vmset.m v8
# CHECK-INST: vmset.m v8
# CHECK-ENCODING: [0x57,0x24,0x84,0x7e]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 24 84 7e <unknown>

vmnot.m v8, v4
# CHECK-INST: vmnot.m v8, v4
# CHECK-ENCODING: [0x57,0x24,0x42,0x76]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 24 42 76 <unknown>
