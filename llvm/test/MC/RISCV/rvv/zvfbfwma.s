# RUN: llvm-mc -triple=riscv32 -show-encoding -mattr=+experimental-zvfbfwma %s \
# RUN:   | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: not llvm-mc -triple=riscv32 -show-encoding -mattr=+v,+f %s 2>&1 \
# RUN:   | FileCheck %s --check-prefix=CHECK-ERROR
# RUN: llvm-mc -triple=riscv32 -filetype=obj -mattr=+experimental-zvfbfwma %s \
# RUN:    | llvm-objdump -d --mattr=+experimental-zvfbfwma --no-print-imm-hex - \
# RUN:    | FileCheck %s --check-prefix=CHECK-INST
# RUN: llvm-mc -triple=riscv32 -filetype=obj -mattr=+experimental-zvfbfwma %s \
# RUN:    | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN
# RUN: llvm-mc -triple=riscv64 -show-encoding -mattr=+experimental-zvfbfwma %s \
# RUN:   | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: not llvm-mc -triple=riscv64 -show-encoding -mattr=+v,+f %s 2>&1 \
# RUN:   | FileCheck %s --check-prefix=CHECK-ERROR
# RUN: llvm-mc -triple=riscv64 -filetype=obj -mattr=+experimental-zvfbfwma %s \
# RUN:    | llvm-objdump -d --mattr=+experimental-zvfbfwma --no-print-imm-hex - \
# RUN:    | FileCheck %s --check-prefix=CHECK-INST
# RUN: llvm-mc -triple=riscv64 -filetype=obj -mattr=+experimental-zvfbfwma %s \
# RUN:    | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

# CHECK-INST: vfwmaccbf16.vv v8, v20, v4, v0.t
# CHECK-ENCODING: [0x57,0x14,0x4a,0xec]
# CHECK-ERROR: instruction requires the following: 'Zvfbfwma' (Vector BF16 widening mul-add){{$}}
# CHECK-UNKNOWN: 57 14 4a ec <unknown>
vfwmaccbf16.vv v8, v20, v4, v0.t

# CHECK-INST: vfwmaccbf16.vv v8, v20, v4
# CHECK-ENCODING: [0x57,0x14,0x4a,0xee]
# CHECK-ERROR: instruction requires the following: 'Zvfbfwma' (Vector BF16 widening mul-add){{$}}
# CHECK-UNKNOWN: 57 14 4a ee <unknown>
vfwmaccbf16.vv v8, v20, v4

# CHECK-INST: vfwmaccbf16.vf v8, fa0, v4, v0.t
# CHECK-ENCODING: [0x57,0x54,0x45,0xec]
# CHECK-ERROR: instruction requires the following: 'Zvfbfwma' (Vector BF16 widening mul-add){{$}}
# CHECK-UNKNOWN: 57 54 45 ec <unknown>
vfwmaccbf16.vf v8, fa0, v4, v0.t

# CHECK-INST: vfwmaccbf16.vf v8, fa0, v4
# CHECK-ENCODING: [0x57,0x54,0x45,0xee]
# CHECK-ERROR: instruction requires the following: 'Zvfbfwma' (Vector BF16 widening mul-add){{$}}
# CHECK-UNKNOWN: 57 54 45 ee <unknown>
vfwmaccbf16.vf v8, fa0, v4

# Check scalar half FP load/store/move included in this extension.

# CHECK-INST: flh ft0, 12(a0)
# CHECK-ENCODING: [0x07,0x10,0xc5,0x00]
# CHECK-ERROR: instruction requires the following: 'Zfh' (Half-Precision Floating-Point) or 'Zfhmin' (Half-Precision Floating-Point Minimal) or 'Zfbfmin' (Scalar BF16 Converts){{$}}
# CHECK-UNKNOWN: 07 10 c5 00 <unknown>
flh f0, 12(a0)

# CHECK-INST: fsh ft6, 2047(s4)
# CHECK-ENCODING: [0xa7,0x1f,0x6a,0x7e]
# CHECK-ERROR: instruction requires the following: 'Zfh' (Half-Precision Floating-Point) or 'Zfhmin' (Half-Precision Floating-Point Minimal) or 'Zfbfmin' (Scalar BF16 Converts){{$}}
# CHECK-UNKNOWN: a7 1f 6a 7e <unknown>
fsh f6, 2047(s4)

# CHECK-INST: fmv.x.h a2, fs7
# CHECK-ENCODING: [0x53,0x86,0x0b,0xe4]
# CHECK-ERROR: instruction requires the following: 'Zfh' (Half-Precision Floating-Point) or 'Zfhmin' (Half-Precision Floating-Point Minimal) or 'Zfbfmin' (Scalar BF16 Converts){{$}}
# CHECK-UNKNOWN: 53 86 0b e4 <unknown>
fmv.x.h a2, fs7

# CHECK-INST: fmv.h.x ft1, a6
# CHECK-ENCODING: [0xd3,0x00,0x08,0xf4]
# CHECK-ERROR: instruction requires the following: 'Zfh' (Half-Precision Floating-Point) or 'Zfhmin' (Half-Precision Floating-Point Minimal) or 'Zfbfmin' (Scalar BF16 Converts){{$}}
# CHECK-UNKNOWN: d3 00 08 f4 <unknown>
fmv.h.x ft1, a6
