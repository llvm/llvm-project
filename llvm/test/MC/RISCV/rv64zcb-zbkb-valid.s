# RUN: llvm-mc %s -triple=riscv64 -mattr=+zbb,+zbkb,+zcb -M no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+zbb,+zbkb,+zcb < %s \
# RUN:     | llvm-objdump --mattr=+zbb,+zbkb,+zcb --no-print-imm-hex -M no-aliases -d -r - \
# RUN:     | FileCheck --check-prefixes=CHECK-ASM-AND-OBJ %s

# Make sure packw spelling of zext.h compresses when Zbb is enabled.

# CHECK-ASM-AND-OBJ: c.zext.h s0
# CHECK-ASM: encoding: [0x69,0x9c]
packw s0, s0, zero
