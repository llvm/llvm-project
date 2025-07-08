# RUN: llvm-mc %s -triple=riscv32 -mattr=+experimental-xrivosvizip -M no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+experimental-xrivosvizip < %s \
# RUN:     | llvm-objdump --mattr=+experimental-xrivosvizip -M no-aliases -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc %s -triple=riscv64 -mattr=+experimental-xrivosvizip -M no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+experimental-xrivosvizip < %s \
# RUN:     | llvm-objdump --mattr=+experimental-xrivosvizip -M no-aliases -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s

# CHECK-ASM-AND-OBJ: ri.vzipeven.vv    v1, v2, v3
# CHECK-ASM: encoding: [0xdb,0x80,0x21,0x32]
ri.vzipeven.vv v1, v2, v3
# CHECK-ASM-AND-OBJ: ri.vzipeven.vv    v1, v2, v3, v0.t
# CHECK-ASM: encoding: [0xdb,0x80,0x21,0x30]
ri.vzipeven.vv v1, v2, v3, v0.t
# CHECK-ASM-AND-OBJ: ri.vzipodd.vv  v1, v2, v3
# CHECK-ASM: encoding: [0xdb,0x80,0x21,0x72]
ri.vzipodd.vv v1, v2, v3
# CHECK-ASM-AND-OBJ: ri.vzipodd.vv  v1, v2, v3, v0.t
# CHECK-ASM: encoding: [0xdb,0x80,0x21,0x70]
ri.vzipodd.vv v1, v2, v3, v0.t

# CHECK-ASM-AND-OBJ:  ri.vzip2a.vv   v1, v2, v3
# CHECK-ASM: encoding: [0xdb,0x80,0x21,0x12]
ri.vzip2a.vv v1, v2, v3
# CHECK-ASM-AND-OBJ:  ri.vzip2a.vv   v1, v2, v3, v0.t
# CHECK-ASM: encoding: [0xdb,0x80,0x21,0x10]
ri.vzip2a.vv v1, v2, v3, v0.t
# CHECK-ASM-AND-OBJ: ri.vzip2b.vv   v1, v2, v3
# CHECK-ASM: encoding: [0xdb,0x80,0x21,0x52]
ri.vzip2b.vv v1, v2, v3
# CHECK-ASM-AND-OBJ: ri.vzip2b.vv   v1, v2, v3, v0.t
# CHECK-ASM: encoding: [0xdb,0x80,0x21,0x50]
ri.vzip2b.vv v1, v2, v3, v0.t

# CHECK-ASM-AND-OBJ: ri.vunzip2a.vv v1, v2, v3
# CHECK-ASM: encoding: [0xdb,0x80,0x21,0x22]
ri.vunzip2a.vv v1, v2, v3
# CHECK-ASM-AND-OBJ: ri.vunzip2a.vv v1, v2, v3, v0.t
# CHECK-ASM: encoding: [0xdb,0x80,0x21,0x20]
ri.vunzip2a.vv v1, v2, v3, v0.t
# CHECK-ASM-AND-OBJ: ri.vunzip2b.vv v1, v2, v3
# CHECK-ASM: encoding: [0xdb,0x80,0x21,0x62]
ri.vunzip2b.vv v1, v2, v3
# CHECK-ASM-AND-OBJ: ri.vunzip2b.vv v1, v2, v3, v0.t
# CHECK-ASM: encoding: [0xdb,0x80,0x21,0x60]
ri.vunzip2b.vv v1, v2, v3, v0.t

# Overlap between source registers *is* allowed

# CHECK-ASM-AND-OBJ: ri.vzipeven.vv    v1, v2, v2
# CHECK-ASM: encoding: [0xdb,0x00,0x21,0x32]
ri.vzipeven.vv v1, v2, v2

# CHECK-ASM-AND-OBJ: ri.vzipeven.vv    v1, v2, v0, v0.t
# CHECK-ASM: encoding: [0xdb,0x00,0x20,0x30]
ri.vzipeven.vv v1, v2, v0, v0.t
