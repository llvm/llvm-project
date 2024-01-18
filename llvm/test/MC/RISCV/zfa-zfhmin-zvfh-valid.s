# RUN: llvm-mc %s -triple=riscv32 -mattr=+zfa,+zfhmin,+zvfh -riscv-no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc %s -triple=riscv64 -mattr=+zfa,+zfhmin,+zvfh -riscv-no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+zfa,+zfhmin,+zvfh < %s \
# RUN:     | llvm-objdump --mattr=+zfa,+zfhmin,+zvfh -M no-aliases -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+zfa,+zfhmin,+zvfh < %s \
# RUN:     | llvm-objdump --mattr=+zfa,+zfhmin,+zvfh -M no-aliases -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s
#
# RUN: not llvm-mc -triple riscv32 -riscv-no-aliases -show-encoding < %s 2>&1 \
# RUN:     | FileCheck -check-prefixes=CHECK-NO-EXT %s
# RUN: not llvm-mc -triple riscv64 -riscv-no-aliases -show-encoding < %s 2>&1 \
# RUN:     | FileCheck -check-prefixes=CHECK-NO-EXT %s

# This test makes sure fli.h is supported with Zvfh.

# CHECK-ASM-AND-OBJ: fli.h ft1, -1.0
# CHECK-ASM: encoding: [0xd3,0x00,0x10,0xf4]
# CHECK-NO-EXT: error: instruction requires the following: 'Zfa' (Additional Floating-Point), 'Zfh' (Half-Precision Floating-Point) or 'Zvfh' (Vector Half-Precision Floating-Point){{$}}
fli.h ft1, -1.0
