# RUN: llvm-mc -triple=ve %s -o - | FileCheck %s
# RUN: llvm-mc -triple=ve -filetype=obj %s -o - | llvm-objdump -r - | FileCheck %s --check-prefix=CHECK-OBJ

    lea %s34, x@tpoff_lo
    and %s34, %s34, (32)0
    lea.sl %s34, x@tpoff_hi(%s34)
    adds.l %s0, %s14, %s34
# CHECK: lea %s34, x@tpoff_lo
# CHECK-NEXT: and %s34, %s34, (32)0
# CHECK-NEXT: lea.sl %s34, x@tpoff_hi(%s34)
# CHECK-NEXT: adds.l %s0, %s14, %s34

# CHECK-OBJ: 0 R_VE_TPOFF_LO32 x
# CHECK-OBJ-NEXT: 10 R_VE_TPOFF_HI32 x
