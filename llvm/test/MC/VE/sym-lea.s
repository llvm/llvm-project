# RUN: llvm-mc -triple=ve %s -o - | FileCheck %s
# RUN: llvm-mc -triple=ve -filetype=obj %s -o - | llvm-objdump -r - | FileCheck %s --check-prefix=CHECK-OBJ

        lea %s0, tgt
        lea %s0, tgt2-.
# CHECK: lea %s0, tgt
# CHECK-NEXT: [[LABEL:.*]]:
# CHECK-NEXT: lea %s0, tgt2-[[LABEL]]

# CHECK-OBJ: 0 R_VE_REFLONG tgt
# CHECK-OBJ: 8 R_VE_SREL32 tgt2
