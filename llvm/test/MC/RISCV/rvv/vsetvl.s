# RUN: llvm-mc -triple=riscv64 -show-encoding --mattr=+experimental-v %s \
# RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: not llvm-mc -triple=riscv64 -show-encoding %s 2>&1 \
# RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+experimental-v %s \
# RUN:        | llvm-objdump -d --mattr=+experimental-v - \
# RUN:        | FileCheck %s --check-prefix=CHECK-INST
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+experimental-v %s \
# RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

vsetvli a2, a0, e32,m1,ta,ma
# CHECK-INST: vsetvli a2, a0, e32,m1,ta,ma
# CHECK-ENCODING: [0x57,0x76,0x85,0x0c]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 76 85 0c <unknown>

vsetvli a2, a0, e32,m2,ta,ma
# CHECK-INST: vsetvli a2, a0, e32,m2,ta,ma
# CHECK-ENCODING: [0x57,0x76,0x95,0x0c]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 76 95 0c <unknown>

vsetvli a2, a0, e32,m4,ta,ma
# CHECK-INST: vsetvli a2, a0, e32,m4,ta,ma
# CHECK-ENCODING: [0x57,0x76,0xa5,0x0c]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 76 a5 0c <unknown>

vsetvli a2, a0, e32,m8,ta,ma
# CHECK-INST: vsetvli a2, a0, e32,m8,ta,ma
# CHECK-ENCODING: [0x57,0x76,0xb5,0x0c]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 76 b5 0c <unknown>

vsetvli a2, a0, e32,mf2,ta,ma
# CHECK-INST: vsetvli a2, a0, e32,mf2,ta,ma
# CHECK-ENCODING: [0x57,0x76,0xb5,0x0e]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 76 b5 0e <unknown>

vsetvli a2, a0, e32,mf4,ta,ma
# CHECK-INST: vsetvli a2, a0, e32,mf4,ta,ma
# CHECK-ENCODING: [0x57,0x76,0xa5,0x0e]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 76 a5 0e <unknown>

vsetvli a2, a0, e32,mf8,ta,ma
# CHECK-INST: vsetvli a2, a0, e32,mf8,ta,ma
# CHECK-ENCODING: [0x57,0x76,0x95,0x0e]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 76 95 0e <unknown>

vsetvli a2, a0, e32,m1,ta,ma
# CHECK-INST: vsetvli a2, a0, e32,m1,ta,ma
# CHECK-ENCODING: [0x57,0x76,0x85,0x0c]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 76 85 0c <unknown>

vsetvli a2, a0, e32,m1,tu,ma
# CHECK-INST: vsetvli a2, a0, e32,m1,tu,ma
# CHECK-ENCODING: [0x57,0x76,0x85,0x08]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 76 85 08 <unknown>

vsetvli a2, a0, e32,m1,ta,mu
# CHECK-INST: vsetvli a2, a0, e32,m1,ta,mu
# CHECK-ENCODING: [0x57,0x76,0x85,0x04]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 76 85 04 <unknown>

vsetvli a2, a0, e32,m1,tu,mu
# CHECK-INST: vsetvli a2, a0, e32,m1
# CHECK-ENCODING: [0x57,0x76,0x85,0x00]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 76 85 00 <unknown>

vsetvl a2, a0, a1
# CHECK-INST: vsetvl a2, a0, a1
# CHECK-ENCODING: [0x57,0x76,0xb5,0x80]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 76 b5 80 <unknown>
