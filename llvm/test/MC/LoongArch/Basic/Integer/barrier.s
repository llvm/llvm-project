## Test valid barrier instructions.

# RUN: llvm-mc %s --triple=loongarch32 --show-encoding \
# RUN:     | FileCheck --check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc %s --triple=loongarch64 --show-encoding \
# RUN:     | FileCheck --check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc %s --triple=loongarch32 --filetype=obj | llvm-objdump -d - \
# RUN:     | FileCheck --check-prefixes=CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc %s --triple=loongarch64 --filetype=obj | llvm-objdump -d - \
# RUN:     | FileCheck --check-prefixes=CHECK-ASM-AND-OBJ %s

# CHECK-ASM-AND-OBJ: dbar 0
# CHECK-ASM: encoding: [0x00,0x00,0x72,0x38]
dbar 0

# CHECK-ASM-AND-OBJ: ibar 0
# CHECK-ASM: encoding: [0x00,0x80,0x72,0x38]
ibar 0

