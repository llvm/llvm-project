# RUN: llvm-mc %s -triple=riscv32 -mattr=zcmp -M no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=zcmp < %s \
# RUN:     | llvm-objdump --mattr=-c,zcmp -M no-aliases -d -r - \
# RUN:     | FileCheck --check-prefixes=CHECK-ASM-AND-OBJ %s

# CHECK-ASM-AND-OBJ: cm.mvsa01 s1, s0
# CHECK-ASM: encoding: [0xa2,0xac]
cm.mvsa01 s1, s0

# CHECK-ASM-AND-OBJ: cm.mva01s s1, s0
# CHECK-ASM: encoding: [0xe2,0xac]
cm.mva01s s1, s0

# CHECK-ASM-AND-OBJ: cm.mva01s s0, s0
# CHECK-ASM: encoding: [0x62,0xac] 
cm.mva01s s0, s0

# CHECK-ASM-AND-OBJ: cm.popret   {ra}, 16
# CHECK-ASM: encoding: [0x42,0xbe]
cm.popret {ra}, 16

# CHECK-ASM-AND-OBJ: cm.popret   {ra}, 16
# CHECK-ASM: encoding: [0x42,0xbe]
cm.popret {x1}, 16

# CHECK-ASM-AND-OBJ: cm.popret   {ra}, 32
# CHECK-ASM: encoding: [0x46,0xbe]
cm.popret {ra}, 32

# CHECK-ASM-AND-OBJ: cm.popret   {ra}, 32
# CHECK-ASM: encoding: [0x46,0xbe]
cm.popret {x1}, 32

# CHECK-ASM-AND-OBJ: cm.popret   {ra, s0}, 64
# CHECK-ASM: encoding: [0x5e,0xbe]
cm.popret {ra, s0}, 64

# CHECK-ASM-AND-OBJ: cm.popret   {ra, s0}, 64
# CHECK-ASM: encoding: [0x5e,0xbe]
cm.popret {x1, x8}, 64

# CHECK-ASM-AND-OBJ: cm.popret   {ra, s0-s1}, 16
# CHECK-ASM: encoding: [0x62,0xbe]
cm.popret {ra,s0-s1}, 16

# CHECK-ASM-AND-OBJ: cm.popret   {ra, s0-s1}, 16
# CHECK-ASM: encoding: [0x62,0xbe]
cm.popret {x1, x8-x9}, 16

# CHECK-ASM-AND-OBJ: cm.popret   {ra, s0-s2}, 32
# CHECK-ASM: encoding: [0x76,0xbe]
cm.popret {ra, s0-s2}, 32

# CHECK-ASM-AND-OBJ: cm.popret   {ra, s0-s2}, 32
# CHECK-ASM: encoding: [0x76,0xbe]
cm.popret {x1, x8-x9, x18}, 32

# CHECK-ASM-AND-OBJ: cm.popret   {ra, s0-s3}, 32
# CHECK-ASM: encoding: [0x82,0xbe]
cm.popret {ra, s0-s3}, 32

# CHECK-ASM-AND-OBJ: cm.popret   {ra, s0-s3}, 32
# CHECK-ASM: encoding: [0x82,0xbe]
cm.popret {x1, x8-x9, x18-x19}, 32

# CHECK-ASM-AND-OBJ: cm.popret   {ra, s0-s5}, 32
# CHECK-ASM: encoding: [0xa2,0xbe]
cm.popret {ra, s0-s5}, 32

# CHECK-ASM-AND-OBJ: cm.popret   {ra, s0-s5}, 32
# CHECK-ASM: encoding: [0xa2,0xbe]
cm.popret {x1, x8-x9, x18-x21}, 32

# CHECK-ASM-AND-OBJ: cm.popret   {ra, s0-s7}, 48
# CHECK-ASM: encoding: [0xc2,0xbe]
cm.popret {ra, s0-s7}, 48

# CHECK-ASM-AND-OBJ: cm.popret   {ra, s0-s7}, 48
# CHECK-ASM: encoding: [0xc2,0xbe]
cm.popret {x1, x8-x9, x18-x23}, 48

# CHECK-ASM-AND-OBJ: cm.popret   {ra, s0-s11}, 112
# CHECK-ASM: encoding: [0xfe,0xbe]
cm.popret {ra, s0-s11}, 112

# CHECK-ASM-AND-OBJ: cm.popret   {ra, s0-s11}, 112
# CHECK-ASM: encoding: [0xfe,0xbe]
cm.popret {x1, x8-x9, x18-x27}, 112

# CHECK-ASM-AND-OBJ: cm.popretz   {ra}, 16
# CHECK-ASM: encoding: [0x42,0xbc]
cm.popretz {ra}, 16

# CHECK-ASM-AND-OBJ: cm.popretz   {ra}, 16
# CHECK-ASM: encoding: [0x42,0xbc]
cm.popretz {x1}, 16

# CHECK-ASM-AND-OBJ: cm.popretz   {ra}, 32
# CHECK-ASM: encoding: [0x46,0xbc]
cm.popretz {ra}, 32

# CHECK-ASM-AND-OBJ: cm.popretz   {ra}, 32
# CHECK-ASM: encoding: [0x46,0xbc]
cm.popretz {x1}, 32

# CHECK-ASM-AND-OBJ: cm.popretz   {ra, s0}, 64
# CHECK-ASM: encoding: [0x5e,0xbc]
cm.popretz {ra, s0}, 64

# CHECK-ASM-AND-OBJ: cm.popretz   {ra, s0}, 64
# CHECK-ASM: encoding: [0x5e,0xbc]
cm.popretz {x1, x8}, 64

# CHECK-ASM-AND-OBJ: cm.popretz   {ra, s0-s1}, 16
# CHECK-ASM: encoding: [0x62,0xbc]
cm.popretz {ra, s0-s1}, 16

# CHECK-ASM-AND-OBJ: cm.popretz   {ra, s0-s1}, 16
# CHECK-ASM: encoding: [0x62,0xbc]
cm.popretz {x1, x8-x9}, 16

# CHECK-ASM-AND-OBJ: cm.popretz   {ra, s0-s2}, 32
# CHECK-ASM: encoding: [0x76,0xbc]
cm.popretz {ra, s0-s2}, 32

# CHECK-ASM-AND-OBJ: cm.popretz   {ra, s0-s2}, 32
# CHECK-ASM: encoding: [0x76,0xbc]
cm.popretz {x1, x8-x9, x18}, 32

# CHECK-ASM-AND-OBJ: cm.popretz   {ra, s0-s3}, 32
# CHECK-ASM: encoding: [0x82,0xbc]
cm.popretz {ra, s0-s3}, 32

# CHECK-ASM-AND-OBJ: cm.popretz   {ra, s0-s3}, 32
# CHECK-ASM: encoding: [0x82,0xbc]
cm.popretz {x1, x8-x9, x18-x19}, 32

# CHECK-ASM-AND-OBJ: cm.popretz   {ra, s0-s5}, 32
# CHECK-ASM: encoding: [0xa2,0xbc]
cm.popretz {ra, s0-s5}, 32

# CHECK-ASM-AND-OBJ: cm.popretz   {ra, s0-s5}, 32
# CHECK-ASM: encoding: [0xa2,0xbc]
cm.popretz {x1, x8-x9, x18-x21}, 32

# CHECK-ASM-AND-OBJ: cm.popretz   {ra, s0-s7}, 48
# CHECK-ASM: encoding: [0xc2,0xbc]
cm.popretz {ra, s0-s7}, 48

# CHECK-ASM-AND-OBJ: cm.popretz   {ra, s0-s7}, 48
# CHECK-ASM: encoding: [0xc2,0xbc]
cm.popretz {x1, x8-x9, x18-x23}, 48

# CHECK-ASM-AND-OBJ: cm.popretz   {ra, s0-s11}, 112
# CHECK-ASM: encoding: [0xfe,0xbc]
cm.popretz {ra, s0-s11}, 112

# CHECK-ASM-AND-OBJ: cm.popretz   {ra, s0-s11}, 112
# CHECK-ASM: encoding: [0xfe,0xbc]
cm.popretz {x1, x8-x9, x18-x27}, 112

# CHECK-ASM-AND-OBJ: cm.pop  {ra}, 16
# CHECK-ASM: encoding: [0x42,0xba]
cm.pop {ra}, 16

# CHECK-ASM-AND-OBJ: cm.pop  {ra}, 16
# CHECK-ASM: encoding: [0x42,0xba]
cm.pop {x1}, 16

# CHECK-ASM-AND-OBJ: cm.pop  {ra}, 32
# CHECK-ASM: encoding: [0x46,0xba]
cm.pop {ra}, 32

# CHECK-ASM-AND-OBJ: cm.pop  {ra}, 32
# CHECK-ASM: encoding: [0x46,0xba]
cm.pop {x1}, 32

# CHECK-ASM-AND-OBJ: cm.pop  {ra, s0}, 16
# CHECK-ASM: encoding: [0x52,0xba]
cm.pop {ra, s0}, 16

# CHECK-ASM-AND-OBJ: cm.pop  {ra, s0}, 16
# CHECK-ASM: encoding: [0x52,0xba]
cm.pop {x1, x8}, 16

# CHECK-ASM-AND-OBJ: cm.pop  {ra, s0-s1}, 32
# CHECK-ASM: encoding: [0x66,0xba]
cm.pop {ra, s0-s1}, 32

# CHECK-ASM-AND-OBJ: cm.pop  {ra, s0-s1}, 32
# CHECK-ASM: encoding: [0x66,0xba]
cm.pop {x1, x8-x9}, 32

# CHECK-ASM-AND-OBJ: cm.pop  {ra, s0-s2}, 32
# CHECK-ASM: encoding: [0x76,0xba]
cm.pop {ra, s0-s2}, 32

# CHECK-ASM-AND-OBJ: cm.pop  {ra, s0-s2}, 32
# CHECK-ASM: encoding: [0x76,0xba]
cm.pop {x1, x8-x9, x18}, 32

# CHECK-ASM-AND-OBJ: cm.pop  {ra, s0-s5}, 32
# CHECK-ASM: encoding: [0xa2,0xba]
cm.pop {ra, s0-s5}, 32

# CHECK-ASM-AND-OBJ: cm.pop  {ra, s0-s5}, 32
# CHECK-ASM: encoding: [0xa2,0xba]
cm.pop {x1, x8-x9, x18-x21}, 32

# CHECK-ASM-AND-OBJ: cm.pop  {ra, s0-s7}, 48
# CHECK-ASM: encoding: [0xc2,0xba]
cm.pop {ra, s0-s7}, 48

# CHECK-ASM-AND-OBJ: cm.pop  {ra, s0-s7}, 48
# CHECK-ASM: encoding: [0xc2,0xba]
cm.pop {x1, x8-x9, x18-x23}, 48

# CHECK-ASM-AND-OBJ: cm.pop  {ra, s0-s11}, 64
# CHECK-ASM: encoding: [0xf2,0xba]
cm.pop {ra, s0-s11}, 64

# CHECK-ASM-AND-OBJ: cm.pop  {ra, s0-s11}, 64
# CHECK-ASM: encoding: [0xf2,0xba]
cm.pop {x1, x8-x9, x18-x27}, 64

# CHECK-ASM-AND-OBJ: cm.push {ra}, -16
# CHECK-ASM: encoding: [0x42,0xb8]
cm.push {ra}, -16

# CHECK-ASM-AND-OBJ: cm.push {ra}, -16
# CHECK-ASM: encoding: [0x42,0xb8]
cm.push {x1}, -16

# CHECK-ASM-AND-OBJ: cm.push {ra, s0}, -32
# CHECK-ASM: encoding: [0x56,0xb8]
cm.push {ra, s0}, -32

# CHECK-ASM-AND-OBJ: cm.push {ra, s0}, -32
# CHECK-ASM: encoding: [0x56,0xb8]
cm.push {x1, x8}, -32

# CHECK-ASM-AND-OBJ: cm.push {ra, s0-s1}, -16
# CHECK-ASM: encoding: [0x62,0xb8]
cm.push {ra, s0-s1}, -16

# CHECK-ASM-AND-OBJ: cm.push {ra, s0-s1}, -16
# CHECK-ASM: encoding: [0x62,0xb8]
cm.push {x1, x8-x9}, -16

# CHECK-ASM-AND-OBJ: cm.push {ra, s0-s3}, -32
# CHECK-ASM: encoding: [0x82,0xb8]
cm.push {ra, s0-s3}, -32

# CHECK-ASM-AND-OBJ: cm.push {ra, s0-s3}, -32
# CHECK-ASM: encoding: [0x82,0xb8]
cm.push {x1, x8-x9, x18-x19}, -32

# CHECK-ASM-AND-OBJ: cm.push {ra, s0-s7}, -48
# CHECK-ASM: encoding: [0xc2,0xb8]
cm.push {ra, s0-s7}, -48

# CHECK-ASM-AND-OBJ: cm.push {ra, s0-s7}, -48
# CHECK-ASM: encoding: [0xc2,0xb8]
cm.push {x1, x8-x9, x18-x23}, -48

# CHECK-ASM-AND-OBJ: cm.push {ra, s0-s7}, -64
# CHECK-ASM: encoding: [0xc6,0xb8]
cm.push {ra, s0-s7}, -64

# CHECK-ASM-AND-OBJ: cm.push {ra, s0-s7}, -64
# CHECK-ASM: encoding: [0xc6,0xb8]
cm.push {x1, x8-x9, x18-x23}, -64

# CHECK-ASM-AND-OBJ: cm.push {ra, s0-s11}, -80
# CHECK-ASM: encoding: [0xf6,0xb8]
cm.push {ra, s0-s11}, -80

# CHECK-ASM-AND-OBJ: cm.push {ra, s0-s11}, -80
# CHECK-ASM: encoding: [0xf6,0xb8]
cm.push {x1, x8-x9, x18-x27}, -80

# CHECK-ASM-AND-OBJ: cm.push {ra, s0-s11}, -112
# CHECK-ASM: encoding: [0xfe,0xb8]
cm.push {ra, s0-s11}, -112

# CHECK-ASM-AND-OBJ: cm.push {ra, s0-s11}, -112
# CHECK-ASM: encoding: [0xfe,0xb8]
cm.push {x1, x8-x9, x18-x27}, -112
