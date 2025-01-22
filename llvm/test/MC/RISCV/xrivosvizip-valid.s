# RUN: llvm-mc %s -triple=riscv32 -mattr=+experimental-xrivosvizip -riscv-no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+experimental-xrivosvizip < %s \
# RUN:     | llvm-objdump --mattr=+experimental-xrivosvizip -M no-aliases -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc %s -triple=riscv64 -mattr=+experimental-xrivosvizip -riscv-no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+experimental-xrivosvizip < %s \
# RUN:     | llvm-objdump --mattr=+experimental-xrivosvizip -M no-aliases -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s

# CHECK-ASM-AND-OBJ: rv.vzipeven.vv    v1, v2, v3
# CHECK-ASM: encoding: [0xdb,0x80,0x21,0x32]
rv.vzipeven.vv v1, v2, v3
# CHECK-ASM-AND-OBJ: rv.vzipodd.vv  v1, v2, v3
# CHECK-ASM: encoding: [0xdb,0x80,0x21,0x72]
rv.vzipodd.vv v1, v2, v3
# CHECK-ASM-AND-OBJ:  rv.vzip2a.vv   v1, v2, v3
# CHECK-ASM: encoding: [0xdb,0x80,0x21,0x12]
rv.vzip2a.vv v1, v2, v3
# CHECK-ASM-AND-OBJ: rv.vzip2b.vv   v1, v2, v3
# CHECK-ASM: encoding: [0xdb,0x80,0x21,0x52]
rv.vzip2b.vv v1, v2, v3
# CHECK-ASM-AND-OBJ: rv.vunzip2a.vv v1, v2, v3
# CHECK-ASM: encoding: [0xdb,0x80,0x21,0x22]
rv.vunzip2a.vv v1, v2, v3
# CHECK-ASM-AND-OBJ: rv.vunzip2b.vv v1, v2, v3
# CHECK-ASM: encoding: [0xdb,0x80,0x21,0x62]
rv.vunzip2b.vv v1, v2, v3

