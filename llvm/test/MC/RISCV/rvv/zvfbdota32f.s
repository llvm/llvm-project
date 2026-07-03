# RUN: llvm-mc -triple=riscv64 -show-encoding -mattr=+experimental-zvfbdota32f %s \
# RUN:   | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: not llvm-mc -triple=riscv64 -show-encoding -mattr=+v %s 2>&1 \
# RUN:   | FileCheck %s --check-prefix=CHECK-ERROR
# RUN: llvm-mc -triple=riscv64 -filetype=obj -mattr=+experimental-zvfbdota32f %s \
# RUN:    | llvm-objdump -d --mattr=+experimental-zvfbdota32f --no-print-imm-hex - \
# RUN:    | FileCheck %s --check-prefix=CHECK-INST

# CHECK-INST: vfbdota.vv v8, v16, v12, 1
# CHECK-ENCODING: [0x77,0x14,0x16,0xaf]
# CHECK-ERROR: instruction requires the following: 'Zvfbdota32f' (FP32 batched dot-product extension){{$}}
vfbdota.vv v8, v16, v12, 1

# CHECK-INST: vfbdota.vv v8, v16, v12, 2, v0.t
# CHECK-ENCODING: [0x77,0x14,0x26,0xad]
# CHECK-ERROR: instruction requires the following: 'Zvfbdota32f' (FP32 batched dot-product extension){{$}}
vfbdota.vv v8, v16, v12, 2, v0.t
