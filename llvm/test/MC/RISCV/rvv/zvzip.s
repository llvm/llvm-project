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

# CHECK-ASM-AND-OBJ: vzip.vv    v1, v2, v3
# CHECK-ASM: encoding: [0xd7,0xa0,0x21,0xfa]
vzip.vv    v1, v2, v3

# CHECK-ASM-AND-OBJ: vzip.vv    v1, v2, v3, v0.t
# CHECK-ASM: encoding: [0xd7,0xa0,0x21,0xf8]
vzip.vv    v1, v2, v3, v0.t

# CHECK-ASM-AND-OBJ: vpaire.vv  v1, v2, v3
# CHECK-ASM: encoding: [0xd7,0x80,0x21,0x3e]
vpaire.vv  v1, v2, v3

# CHECK-ASM-AND-OBJ: vpaire.vv  v1, v2, v3, v0.t
# CHECK-ASM: encoding: [0xd7,0x80,0x21,0x3c]
vpaire.vv  v1, v2, v3, v0.t

# CHECK-ASM-AND-OBJ:  vpairo.vv   v1, v2, v3
# CHECK-ASM: encoding: [0xd7,0xa0,0x21,0x3e]
vpairo.vv   v1, v2, v3

# CHECK-ASM-AND-OBJ:  vpairo.vv   v1, v2, v3, v0.t
# CHECK-ASM: encoding: [0xd7,0xa0,0x21,0x3c]
vpairo.vv   v1, v2, v3, v0.t

# CHECK-ASM-AND-OBJ: vunzipe.v   v1, v2
# CHECK-ASM: encoding: [0xd7,0xa0,0x25,0x4a]
vunzipe.v   v1, v2

# CHECK-ASM-AND-OBJ: vunzipe.v   v1, v2, v0.t
# CHECK-ASM: encoding: [0xd7,0xa0,0x25,0x48]
vunzipe.v   v1, v2, v0.t

# CHECK-ASM-AND-OBJ: vunzipo.v v1, v2
# CHECK-ASM: encoding: [0xd7,0xa0,0x27,0x4a]
vunzipo.v v1, v2

# CHECK-ASM-AND-OBJ: vunzipe.v   v1, v2, v0.t
# CHECK-ASM: encoding: [0xd7,0xa0,0x25,0x48]
vunzipe.v   v1, v2, v0.t
