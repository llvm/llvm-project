# RUN: llvm-mc %s -triple=riscv32 -mattr=+zfa,+zfhmin,+zvfh -M no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc %s -triple=riscv64 -mattr=+zfa,+zfhmin,+zvfh -M no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+zfa,+zfhmin,+zvfh < %s \
# RUN:     | llvm-objdump --mattr=+zfa,+zfhmin,+zvfh -M no-aliases -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+zfa,+zfhmin,+zvfh < %s \
# RUN:     | llvm-objdump --mattr=+zfa,+zfhmin,+zvfh -M no-aliases -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s
#
# RUN: llvm-mc --verify -triple riscv32 -M no-aliases -show-encoding < %s
# RUN: llvm-mc --verify -triple riscv64 -M no-aliases -show-encoding < %s

# This test makes sure fli.h is supported with Zvfh.

# CHECK-ASM-AND-OBJ: fli.h ft1, -1.0
# CHECK-ASM: encoding: [0xd3,0x00,0x10,0xf4]
fli.h ft1, -1.0 # expected-error {{instruction requires the following: 'Zfa' (Additional Floating-Point), 'Zfh' (Half-Precision Floating-Point) or 'Zvfh' (Vector Half-Precision Floating-Point)}}
