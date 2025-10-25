# RUN: llvm-mc -triple=riscv32 -show-encoding -mattr=+experimental-zvfofp4min %s \
# RUN:   | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: not llvm-mc -triple=riscv32 -show-encoding -mattr=+v,+f %s 2>&1 \
# RUN:   | FileCheck %s --check-prefix=CHECK-ERROR
# RUN: llvm-mc -triple=riscv32 -filetype=obj -mattr=+experimental-zvfofp4min %s \
# RUN:    | llvm-objdump -d --mattr=+experimental-zvfofp4min - \
# RUN:    | FileCheck %s --check-prefix=CHECK-INST
# RUN: llvm-mc -triple=riscv32 -filetype=obj -mattr=+experimental-zvfofp4min %s \
# RUN:    | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN
# RUN: llvm-mc -triple=riscv64 -show-encoding -mattr=+experimental-zvfofp4min %s \
# RUN:   | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: not llvm-mc -triple=riscv64 -show-encoding -mattr=+v,+f %s 2>&1 \
# RUN:   | FileCheck %s --check-prefix=CHECK-ERROR
# RUN: llvm-mc -triple=riscv64 -filetype=obj -mattr=+experimental-zvfofp4min %s \
# RUN:    | llvm-objdump -d --mattr=+experimental-zvfofp4min - \
# RUN:    | FileCheck %s --check-prefix=CHECK-INST
# RUN: llvm-mc -triple=riscv64 -filetype=obj -mattr=+experimental-zvfofp4min %s \
# RUN:    | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

# CHECK-INST: vfext.vf2 v8, v4
# CHECK-ENCODING: [0x57,0x24,0x4b,0x4a]
# CHECK-ERROR: instruction requires the following: 'Zvfofp4min' (Vector OFP4 Converts){{$}}
# CHECK-UNKNOWN: 4a4b2457 <unknown>
vfext.vf2 v8, v4

# CHECK-INST: vfext.vf2 v8, v4, v0.t
# CHECK-ENCODING: [0x57,0x24,0x4b,0x48]
# CHECK-ERROR: instruction requires the following: 'Zvfofp4min' (Vector OFP4 Converts){{$}}
# CHECK-UNKNOWN: 484b2457 <unknown>
vfext.vf2 v8, v4, v0.t
