# RUN: llvm-mc %s --triple=loongarch32 --mattr=+f --show-encoding \
# RUN:     | FileCheck --check-prefixes=ASM-AND-OBJ,ASM %s
# RUN: llvm-mc %s --triple=loongarch64 --mattr=+f --show-encoding \
# RUN:     | FileCheck --check-prefixes=ASM-AND-OBJ,ASM %s
# RUN: llvm-mc %s --triple=loongarch32 --mattr=+f --filetype=obj \
# RUN:     | llvm-objdump -d --mattr=+f - \
# RUN:     | FileCheck --check-prefix=ASM-AND-OBJ %s
# RUN: llvm-mc %s --triple=loongarch64 --mattr=+f --filetype=obj \
# RUN:     | llvm-objdump -d --mattr=+f - \
# RUN:     | FileCheck --check-prefix=ASM-AND-OBJ %s

# ASM-AND-OBJ: bceqz $fcc6, 12
# ASM: encoding: [0xc0,0x0c,0x00,0x48]
bceqz $fcc6, 12

# ASM-AND-OBJ: bcnez $fcc6, 72
# ASM: encoding: [0xc0,0x49,0x00,0x48]
bcnez $fcc6, 72
