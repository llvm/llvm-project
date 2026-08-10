# RUN: llvm-mc -triple=ve %s -o - | FileCheck %s
# RUN: llvm-mc -triple=ve -filetype=obj %s -o - | llvm-objdump -r - | FileCheck %s --check-prefix=CHECK-OBJ

    lea %s15, _GLOBAL_OFFSET_TABLE_@pc_lo(-24)
    and %s15, %s15, (32)0
    sic %s16
    lea.sl %s15, _GLOBAL_OFFSET_TABLE_@pc_hi(%s16, %s15)
    lea %s0, dst@got_lo
    and %s0, %s0, (32)0
    lea.sl %s0, dst@got_hi(, %s0)
    ld %s1, (%s0, %s15)
# CHECK: lea %s15, _GLOBAL_OFFSET_TABLE_@pc_lo(-24)
# CHECK-NEXT: and %s15, %s15, (32)0
# CHECK-NEXT: sic %s16
# CHECK-NEXT: lea.sl %s15, _GLOBAL_OFFSET_TABLE_@pc_hi(%s16, %s15)
# CHECK-NEXT: lea %s0, dst@got_lo
# CHECK-NEXT: and %s0, %s0, (32)0
# CHECK-NEXT: lea.sl %s0, dst@got_hi(, %s0)
# CHECK-NEXT: ld %s1, (%s0, %s15)

# CHECK-OBJ: 0 R_VE_PC_LO32 _GLOBAL_OFFSET_TABLE_
# CHECK-OBJ-NEXT: 18 R_VE_PC_HI32 _GLOBAL_OFFSET_TABLE_
# CHECK-OBJ-NEXT: 20 R_VE_GOT_LO32 dst
# CHECK-OBJ-NEXT: 30 R_VE_GOT_HI32 dst
