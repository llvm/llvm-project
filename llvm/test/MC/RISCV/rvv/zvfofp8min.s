# RUN: llvm-mc -triple=riscv32 -show-encoding -mattr=+experimental-zvfofp8min %s \
# RUN:   | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --verify -triple=riscv32 -show-encoding -mattr=+v,+f %s
# RUN: llvm-mc -triple=riscv32 -filetype=obj -mattr=+experimental-zvfofp8min %s \
# RUN:    | llvm-objdump -d --mattr=+experimental-zvfofp8min - \
# RUN:    | FileCheck %s --check-prefix=CHECK-INST
# RUN: llvm-mc -triple=riscv64 -show-encoding -mattr=+experimental-zvfofp8min %s \
# RUN:   | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --verify -triple=riscv64 -show-encoding -mattr=+v,+f %s
# RUN: llvm-mc -triple=riscv64 -filetype=obj -mattr=+experimental-zvfofp8min %s \
# RUN:    | llvm-objdump -d --mattr=+experimental-zvfofp8min - \
# RUN:    | FileCheck %s --check-prefix=CHECK-INST

# CHECK-INST: vfncvtbf16.sat.f.f.w v8, v4, v0.t
# CHECK-ENCODING: [0x57,0x94,0x4f,0x48]
vfncvtbf16.sat.f.f.w v8, v4, v0.t # expected-error {{instruction requires the following: 'Zvfofp8min' (Vector OFP8 Converts)}}

# CHECK-INST: vfncvt.f.f.q v8, v4, v0.t
# CHECK-ENCODING: [0x57,0x94,0x4c,0x48]
vfncvt.f.f.q v8, v4, v0.t # expected-error {{instruction requires the following: 'Zvfofp8min' (Vector OFP8 Converts)}}

# CHECK-INST: vfncvt.sat.f.f.q v8, v4, v0.t
# CHECK-ENCODING: [0x57,0x94,0x4d,0x48]
vfncvt.sat.f.f.q v8, v4, v0.t # expected-error {{instruction requires the following: 'Zvfofp8min' (Vector OFP8 Converts)}}
