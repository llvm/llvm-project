# RUN: llvm-mc -triple=riscv64 -show-encoding -mattr=+experimental-zvfbdota32f %s \
# RUN:   | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --verify -triple=riscv64 -show-encoding -mattr=+v %s
# RUN: llvm-mc -triple=riscv64 -filetype=obj -mattr=+experimental-zvfbdota32f %s \
# RUN:    | llvm-objdump -d --mattr=+experimental-zvfbdota32f --no-print-imm-hex - \
# RUN:    | FileCheck %s --check-prefix=CHECK-INST

# CHECK-INST: vfbdota.vv v8, v16, v12, 8
# CHECK-ENCODING: [0x77,0x14,0x16,0xaf]
vfbdota.vv v8, v16, v12, 8 # expected-error {{instruction requires the following: 'Zvfbdota32f' (FP32 batched dot-product extension)}}

# CHECK-INST: vfbdota.vv v8, v16, v12, 16, v0.t
# CHECK-ENCODING: [0x77,0x14,0x26,0xad]
vfbdota.vv v8, v16, v12, 16, v0.t # expected-error {{instruction requires the following: 'Zvfbdota32f' (FP32 batched dot-product extension)}}
