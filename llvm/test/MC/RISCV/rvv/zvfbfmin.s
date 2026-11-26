# RUN: llvm-mc -triple=riscv32 -show-encoding -mattr=+zvfbfmin %s \
# RUN:   | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --verify -triple=riscv32 -show-encoding -mattr=+v,+f %s
# RUN: llvm-mc -triple=riscv32 -filetype=obj -mattr=+zvfbfmin %s \
# RUN:    | llvm-objdump -d --mattr=+zvfbfmin - \
# RUN:    | FileCheck %s --check-prefix=CHECK-INST
# RUN: llvm-mc -triple=riscv64 -show-encoding -mattr=+zvfbfmin %s \
# RUN:   | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --verify -triple=riscv64 -show-encoding -mattr=+v,+f %s
# RUN: llvm-mc -triple=riscv64 -filetype=obj -mattr=+zvfbfmin %s \
# RUN:    | llvm-objdump -d --mattr=+zvfbfmin - \
# RUN:    | FileCheck %s --check-prefix=CHECK-INST

# CHECK-INST: vfncvtbf16.f.f.w v8, v4, v0.t
# CHECK-ENCODING: [0x57,0x94,0x4e,0x48]
vfncvtbf16.f.f.w v8, v4, v0.t # expected-error {{instruction requires the following: 'Zvfbfmin' (Vector BF16 Converts) or 'Zvfofp8min' (Vector OFP8 Converts)}}

# CHECK-INST: vfncvtbf16.f.f.w v8, v4
# CHECK-ENCODING: [0x57,0x94,0x4e,0x4a]
vfncvtbf16.f.f.w v8, v4 # expected-error {{instruction requires the following: 'Zvfbfmin' (Vector BF16 Converts) or 'Zvfofp8min' (Vector OFP8 Converts)}}

# CHECK-INST: vfwcvtbf16.f.f.v v8, v4, v0.t
# CHECK-ENCODING: [0x57,0x94,0x46,0x48]
vfwcvtbf16.f.f.v v8, v4, v0.t # expected-error {{instruction requires the following: 'Zvfbfmin' (Vector BF16 Converts) or 'Zvfofp8min' (Vector OFP8 Converts)}}

# CHECK-INST: vfwcvtbf16.f.f.v v8, v4
# CHECK-ENCODING: [0x57,0x94,0x46,0x4a]
vfwcvtbf16.f.f.v v8, v4 # expected-error {{instruction requires the following: 'Zvfbfmin' (Vector BF16 Converts) or 'Zvfofp8min' (Vector OFP8 Converts)}}
