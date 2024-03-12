# RUN: llvm-mc -triple=riscv64 -show-encoding --mattr=+xtheadvector %s \
# RUN:   --mattr=+f --riscv-no-aliases \
# RUN:   | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: not llvm-mc -triple=riscv64 -show-encoding %s 2>&1 \
# RUN:   | FileCheck %s --check-prefix=CHECK-ERROR
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+xtheadvector %s \
# RUN:   --mattr=+f | llvm-objdump -d --mattr=+xtheadvector --mattr=+f -M no-aliases - \
# RUN:   | FileCheck %s --check-prefix=CHECK-INST
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+xtheadvector %s \
# RUN:   --mattr=+f | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

th.vfsqrt.v v8, v4, v0.t
# CHECK-INST: th.vfsqrt.v v8, v4, v0.t
# CHECK-ENCODING: [0x57,0x14,0x40,0x8c]
# CHECK-ERROR: instruction requires the following: 'F' (Single-Precision Floating-Point), 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 14 40 8c <unknown>

th.vfsqrt.v v8, v4
# CHECK-INST: th.vfsqrt.v v8, v4
# CHECK-ENCODING: [0x57,0x14,0x40,0x8e]
# CHECK-ERROR: instruction requires the following: 'F' (Single-Precision Floating-Point), 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 14 40 8e <unknown>

th.vfclass.v v8, v4, v0.t
# CHECK-INST: th.vfclass.v v8, v4, v0.t
# CHECK-ENCODING: [0x57,0x14,0x48,0x8c]
# CHECK-ERROR: instruction requires the following: 'F' (Single-Precision Floating-Point), 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 14 48 8c <unknown>

th.vfclass.v v8, v4
# CHECK-INST: th.vfclass.v v8, v4
# CHECK-ENCODING: [0x57,0x14,0x48,0x8e]
# CHECK-ERROR: instruction requires the following: 'F' (Single-Precision Floating-Point), 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 14 48 8e <unknown>

th.vfmerge.vfm v8, v4, fa0, v0
# CHECK-INST: th.vfmerge.vfm v8, v4, fa0, v0
# CHECK-ENCODING: [0x57,0x54,0x45,0x5c]
# CHECK-ERROR: instruction requires the following: 'F' (Single-Precision Floating-Point), 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 54 45 5c <unknown>
