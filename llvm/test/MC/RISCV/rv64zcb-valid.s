# RUN: llvm-mc %s -triple=riscv64 -mattr=+m,+zbb,+zba,+experimental-zcb -riscv-no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+m,+zbb,+zba,+experimental-zcb < %s \
# RUN:     | llvm-objdump --mattr=+m,+zbb,+zba,experimental-zcb -M no-aliases -d -r - \
# RUN:     | FileCheck --check-prefixes=CHECK-ASM-AND-OBJ %s
#
# RUN: not llvm-mc -triple riscv64 \
# RUN:     -riscv-no-aliases -show-encoding < %s 2>&1 \
# RUN:     | FileCheck -check-prefixes=CHECK-NO-EXT %s
# RUN: not llvm-mc -triple riscv32 -mattr=+m,+zbb,+zba,+experimental-zcb \
# RUN:     -riscv-no-aliases -show-encoding < %s 2>&1 \
# RUN:     | FileCheck -check-prefixes=CHECK-NO-RV64 %s

# CHECK-ASM-AND-OBJ: c.zext.w s0
# CHECK-ASM: encoding: [0x71,0x9c]
# CHECK-NO-EXT: error: instruction requires the following: 'Zba' (Address Generation Instructions), 'Zcb' (Compressed basic bit manipulation instructions){{$}}
# CHECK-NO-RV64: error: instruction requires the following: RV64I Base Instruction Set{{$}}
c.zext.w s0

# CHECK-ASM-AND-OBJ: c.zext.w s0
# CHECK-ASM: encoding: [0x71,0x9c]
add.uw s0, s0, zero
