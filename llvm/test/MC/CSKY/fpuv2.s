# RUN: llvm-mc %s -triple=csky -show-encoding -csky-no-aliases -mattr=+fpuv2_sf -mattr=+fpuv2_df \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s

# CHECK-ASM-AND-OBJ: fldms  vr1-vr2, (a1)
# CHECK-ASM: encoding: [0x21,0xf4,0x01,0x30]
fldms  vr1-vr2, (a1)

# CHECK-ASM-AND-OBJ: fldmd  vr1-vr2, (a1)
# CHECK-ASM: encoding: [0x21,0xf4,0x01,0x31]
fldmd  vr1-vr2, (a1)

# CHECK-ASM-AND-OBJ: fstms  vr1-vr2, (a1)
# CHECK-ASM: encoding: [0x21,0xf4,0x01,0x34]
fstms  vr1-vr2, (a1)

# CHECK-ASM-AND-OBJ: fstmd  vr1-vr2, (a1)
# CHECK-ASM: encoding: [0x21,0xf4,0x01,0x35]
fstmd  vr1-vr2, (a1)

# RUN: not llvm-mc -triple csky -mattr=+fpuv2_sf -mattr=+fpuv2_df --defsym=ERR=1 < %s 2>&1 | FileCheck %s

.ifdef ERR
fldms  vr1-vr33, (a1) # CHECK: :[[#@LINE]]:12: error: invalid register
fldms  vr1-vr31, (a1) # CHECK: :[[#@LINE]]:8: error: Register sequence is not valid
fstms  vr1-vr33, (a1) # CHECK: :[[#@LINE]]:12: error: invalid register
fstms  vr1-vr31, (a1) # CHECK: :[[#@LINE]]:8: error: Register sequence is not valid
fldmd  vr1-vr33, (a1) # CHECK: :[[#@LINE]]:12: error: invalid register
fldmd  vr1-vr31, (a1) # CHECK: :[[#@LINE]]:8: error: Register sequence is not valid
fstmd  vr1-vr33, (a1) # CHECK: :[[#@LINE]]:12: error: invalid register
fstmd  vr1-vr31, (a1) # CHECK: :[[#@LINE]]:8: error: Register sequence is not valid
.endif
