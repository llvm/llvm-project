# RUN: llvm-mc -triple=ve %s -o - | FileCheck %s
# RUN: llvm-mc -triple=ve -filetype=obj %s -o - | llvm-objdump -r - | FileCheck %s --check-prefix=CHECK-OBJ

        lea %s0, var
        lea %s1, var@lo
        and %s1, %s1, (32)0
        lea.sl %s1, var@hi(, %s1)
        lea %s1, var+8@lo
        and %s1, %s1, (32)0
        lea.sl %s1, var+8@hi(, %s1)
# CHECK: lea %s0, var
# CHECK-NEXT: lea %s1, var@lo
# CHECK-NEXT: and %s1, %s1, (32)0
# CHECK-NEXT: lea.sl %s1, var@hi(, %s1)
# CHECK-NEXT: lea %s1, var+8@lo
# CHECK-NEXT: and %s1, %s1, (32)0
# CHECK-NEXT: lea.sl %s1, var+8@hi(, %s1)

# CHECK-OBJ: 0 R_VE_REFLONG var
# CHECK-OBJ-NEXT: 8 R_VE_LO32 var
# CHECK-OBJ-NEXT: 18 R_VE_HI32 var
# CHECK-OBJ-NEXT: 20 R_VE_LO32 var+0x8
# CHECK-OBJ-NEXT: 30 R_VE_HI32 var+0x8
