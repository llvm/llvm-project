# RUN: llvm-mc -triple=ve %s -o - | FileCheck %s
# RUN: llvm-mc -triple=ve -filetype=obj %s -o - | llvm-objdump -r - | FileCheck %s --check-prefix=CHECK-OBJ

        b.l.t tgt
        br.l.t tgt2
        b.l.t tgt(, %s1)
        b.l.t tgt+24(, %s1)
# CHECK: b.l.t tgt
# CHECK-NEXT: br.l.t tgt2
# CHECK-NEXT: b.l.t tgt(, %s1)
# CHECK-NEXT: b.l.t tgt+24(, %s1)

# CHECK-OBJ: 0 R_VE_REFLONG tgt
# CHECK-OBJ: 8 R_VE_SREL32 tgt2
# CHECK-OBJ: 10 R_VE_REFLONG tgt
# CHECK-OBJ: 18 R_VE_REFLONG tgt+0x18
