# RUN: llvm-mc %s -triple=riscv32 -mattr=+experimental-zvzip -riscv-no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc %s -triple=riscv64 -mattr=+experimental-zvzip -riscv-no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+experimental-zvzip < %s \
# RUN:     | llvm-objdump --mattr=+experimental-zvzip -M no-aliases -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+experimental-zvzip < %s \
# RUN:     | llvm-objdump --mattr=+experimental-zvzip -M no-aliases -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s

# CHECK-ASM-AND-OBJ: vzipeven.vv    v1, v2, v3
# CHECK-ASM: encoding: [0xdb,0x80,0x21,0x32]
vzipeven.vv v1, v2, v3
# CHECK-ASM-AND-OBJ: vzipodd.vv  v1, v2, v3
# CHECK-ASM: encoding: [0xdb,0x80,0x21,0x72]
vzipodd.vv v1, v2, v3
# CHECK-ASM-AND-OBJ:  vzip2a.vv   v1, v2, v3
# CHECK-ASM: encoding: [0xdb,0x80,0x21,0x12]
vzip2a.vv v1, v2, v3
# CHECK-ASM-AND-OBJ: vzip2b.vv   v1, v2, v3
# CHECK-ASM: encoding: [0xdb,0x80,0x21,0x52]
vzip2b.vv v1, v2, v3
# CHECK-ASM-AND-OBJ: vunzip2a.vv v1, v2, v3
# CHECK-ASM: encoding: [0xdb,0x80,0x21,0x22]
vunzip2a.vv v1, v2, v3
# CHECK-ASM-AND-OBJ: vunzip2b.vv v1, v2, v3
# CHECK-ASM: encoding: [0xdb,0x80,0x21,0x62]
vunzip2b.vv v1, v2, v3

