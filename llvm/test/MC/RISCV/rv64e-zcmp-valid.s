# RUN: llvm-mc %s -triple=riscv64 -mattr=zcmp,+e -M no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=zcmp < %s \
# RUN:     | llvm-objdump --mattr=-c,zcmp -M no-aliases -d -r - \
# RUN:     | FileCheck --check-prefixes=CHECK-ASM-AND-OBJ %s

# CHECK-ASM-AND-OBJ: cm.mvsa01 s1, s0
# CHECK-ASM: encoding: [0xa2,0xac]
cm.mvsa01 s1, s0

# CHECK-ASM-AND-OBJ: cm.mva01s s1, s0
# CHECK-ASM: encoding: [0xe2,0xac]
cm.mva01s s1, s0

# CHECK-ASM-AND-OBJ: cm.popret   {ra}, 16
# CHECK-ASM: encoding: [0x42,0xbe]
cm.popret {ra}, 16

# CHECK-ASM-AND-OBJ: cm.popret   {ra}, 32
# CHECK-ASM: encoding: [0x46,0xbe]
cm.popret {ra}, 32

# CHECK-ASM-AND-OBJ: cm.popret   {ra, s0}, 64
# CHECK-ASM: encoding: [0x5e,0xbe]
cm.popret {ra, s0}, 64

# CHECK-ASM-AND-OBJ: cm.popret   {ra, s0-s1}, 32
# CHECK-ASM: encoding: [0x62,0xbe]
cm.popret {ra,s0-s1}, 32

# CHECK-ASM-AND-OBJ: cm.popretz   {ra}, 16
# CHECK-ASM: encoding: [0x42,0xbc]
cm.popretz {ra}, 16

# CHECK-ASM-AND-OBJ: cm.popretz   {ra}, 32
# CHECK-ASM: encoding: [0x46,0xbc]
cm.popretz {ra}, 32

# CHECK-ASM-AND-OBJ: cm.popretz   {ra, s0}, 64
# CHECK-ASM: encoding: [0x5e,0xbc]
cm.popretz {ra, s0}, 64

# CHECK-ASM-AND-OBJ: cm.popretz   {ra, s0-s1}, 32
# CHECK-ASM: encoding: [0x62,0xbc]
cm.popretz {ra, s0-s1}, 32

# CHECK-ASM-AND-OBJ: cm.pop  {ra}, 16
# CHECK-ASM: encoding: [0x42,0xba]
cm.pop {ra}, 16

# CHECK-ASM-AND-OBJ: cm.pop  {ra}, 32
# CHECK-ASM: encoding: [0x46,0xba]
cm.pop {ra}, 32

# CHECK-ASM-AND-OBJ: cm.pop  {ra, s0}, 16
# CHECK-ASM: encoding: [0x52,0xba]
cm.pop {ra, s0}, 16

# CHECK-ASM-AND-OBJ: cm.pop  {ra, s0-s1}, 32
# CHECK-ASM: encoding: [0x62,0xba]
cm.pop {ra, s0-s1}, 32

# CHECK-ASM-AND-OBJ: cm.push {ra}, -16
# CHECK-ASM: encoding: [0x42,0xb8]
cm.push {ra}, -16

# CHECK-ASM-AND-OBJ: cm.push {ra, s0}, -32
# CHECK-ASM: encoding: [0x56,0xb8]
cm.push {ra, s0}, -32

# CHECK-ASM-AND-OBJ: cm.push {ra, s0-s1}, -32
# CHECK-ASM: encoding: [0x62,0xb8]
cm.push {ra, s0-s1}, -32
