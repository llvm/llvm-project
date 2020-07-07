# RUN: llvm-mc -triple=ve %s -o - | FileCheck %s
# RUN: llvm-mc -triple=ve -filetype=obj %s -o - | llvm-objdump -r - | FileCheck %s --check-prefix=CHECK-OBJ

        b.l.t tgt(, %s1)
        b.l.t tgt+24(, %s1)
# CHECK: b.l.t tgt(, %s1)
# CHECK-NEXT: b.l.t tgt+24(, %s1)

# CHECK-OBJ: 0 R_VE_REFLONG tgt
# CHECK-OBJ-NEXT: 8 R_VE_REFLONG tgt+0x18
