# XAndesVBFHCvt - Andes Vector BFLOAT16 Conversion Extension
# RUN: llvm-mc %s -triple=riscv32 -mattr=+xandesvbfhcvt -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM %s
# RUN: llvm-mc -filetype=obj -triple riscv32 -mattr=+xandesvbfhcvt < %s \
# RUN:     | llvm-objdump --mattr=+xandesvbfhcvt -M no-aliases -d -r - \
# RUN:     | FileCheck -check-prefixes=CHECK-OBJ %s
# RUN: not llvm-mc -triple=riscv32 -show-encoding %s 2>&1 \
# RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
# RUN: llvm-mc %s -triple=riscv64 -mattr=+xandesvbfhcvt -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM %s
# RUN: llvm-mc -filetype=obj -triple riscv64 -mattr=+xandesvbfhcvt < %s \
# RUN:     | llvm-objdump --mattr=+xandesvbfhcvt -M no-aliases -d -r - \
# RUN:     | FileCheck -check-prefixes=CHECK-OBJ %s
# RUN: not llvm-mc -triple=riscv64 -show-encoding %s 2>&1 \
# RUN:        | FileCheck %s --check-prefix=CHECK-ERROR

# CHECK-OBJ: nds.vfwcvt.s.bf16 v8, v10
# CHECK-ASM: nds.vfwcvt.s.bf16 v8, v10
# CHECK-ASM: encoding: [0x5b,0x44,0xa0,0x00]
# CHECK-ERROR: instruction requires the following: 'XAndesVBFHCvt' (Andes Vector BFLOAT16 Conversion Extension){{$}}
nds.vfwcvt.s.bf16 v8, v10

# CHECK-OBJ: nds.vfncvt.bf16.s v8, v10
# CHECK-ASM: nds.vfncvt.bf16.s v8, v10
# CHECK-ASM: encoding: [0x5b,0xc4,0xa0,0x00]
# CHECK-ERROR: instruction requires the following: 'XAndesVBFHCvt' (Andes Vector BFLOAT16 Conversion Extension){{$}}
nds.vfncvt.bf16.s v8, v10
