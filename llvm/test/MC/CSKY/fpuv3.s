# RUN: llvm-mc %s -triple=csky -show-encoding -csky-no-aliases -mattr=+fpuv3_sf,+fpuv3_df \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s

# CHECK-ASM-AND-OBJ: fldm.32 vr1-vr2, (a1)
# CHECK-ASM: encoding: [0x21,0xf4,0x01,0x30]
fldm.32  vr1-vr2, (a1)

# CHECK-ASM-AND-OBJ: fldm.64 vr1-vr2, (a1)
# CHECK-ASM: encoding: [0x21,0xf4,0x01,0x31]
fldm.64  vr1-vr2, (a1)

# CHECK-ASM-AND-OBJ: fstm.32 vr1-vr2, (a1)
# CHECK-ASM: encoding: [0x21,0xf4,0x01,0x34]
fstm.32 vr1-vr2, (a1)

# CHECK-ASM-AND-OBJ: fstm.64 vr1-vr2, (a1)
# CHECK-ASM: encoding: [0x21,0xf4,0x01,0x35]
fstm.64  vr1-vr2, (a1)

# RUN: not llvm-mc -triple csky -mattr=+fpuv3_sf -mattr=+fpuv3_df --defsym=ERR=1 < %s 2>&1 | FileCheck %s

.ifdef ERR
fstm.32  vr1-vr33, (a1) # CHECK: :[[#@LINE]]:14: error: invalid register
fstm.64  vr1-vr33, (a1) # CHECK: :[[#@LINE]]:14: error: invalid register
fldm.32  vr1-vr33, (a1) # CHECK: :[[#@LINE]]:14: error: invalid register
fldm.64  vr1-vr33, (a1) # CHECK: :[[#@LINE]]:14: error: invalid register
.endif
