# RUN: llvm-mc -triple=riscv32 -show-encoding -mattr=+zvfbfwma %s \
# RUN:   | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: not llvm-mc -triple=riscv32 -show-encoding -mattr=+v,+f %s 2>&1 \
# RUN:   | FileCheck %s --check-prefix=CHECK-ERROR
# RUN: llvm-mc -triple=riscv32 -filetype=obj -mattr=+zvfbfwma %s \
# RUN:    | llvm-objdump -d --mattr=+zvfbfwma --no-print-imm-hex - \
# RUN:    | FileCheck %s --check-prefix=CHECK-INST
# RUN: llvm-mc -triple=riscv64 -show-encoding -mattr=+zvfbfwma %s \
# RUN:   | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: not llvm-mc -triple=riscv64 -show-encoding -mattr=+v,+f %s 2>&1 \
# RUN:   | FileCheck %s --check-prefix=CHECK-ERROR
# RUN: llvm-mc -triple=riscv64 -filetype=obj -mattr=+zvfbfwma %s \
# RUN:    | llvm-objdump -d --mattr=+zvfbfwma --no-print-imm-hex - \
# RUN:    | FileCheck %s --check-prefix=CHECK-INST

# CHECK-INST: vfwmaccbf16.vv v8, v20, v4, v0.t
# CHECK-ENCODING: [0x57,0x14,0x4a,0xec]
# CHECK-ERROR: instruction requires the following: 'Zvfbfwma' (Vector BF16 widening mul-add){{$}}
vfwmaccbf16.vv v8, v20, v4, v0.t

# CHECK-INST: vfwmaccbf16.vv v8, v20, v4
# CHECK-ENCODING: [0x57,0x14,0x4a,0xee]
# CHECK-ERROR: instruction requires the following: 'Zvfbfwma' (Vector BF16 widening mul-add){{$}}
vfwmaccbf16.vv v8, v20, v4

# CHECK-INST: vfwmaccbf16.vf v8, fa0, v4, v0.t
# CHECK-ENCODING: [0x57,0x54,0x45,0xec]
# CHECK-ERROR: instruction requires the following: 'Zvfbfwma' (Vector BF16 widening mul-add){{$}}
vfwmaccbf16.vf v8, fa0, v4, v0.t

# CHECK-INST: vfwmaccbf16.vf v8, fa0, v4
# CHECK-ENCODING: [0x57,0x54,0x45,0xee]
# CHECK-ERROR: instruction requires the following: 'Zvfbfwma' (Vector BF16 widening mul-add){{$}}
vfwmaccbf16.vf v8, fa0, v4
