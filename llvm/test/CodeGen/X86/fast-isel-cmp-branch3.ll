; RUN: llc < %s -fast-isel -fast-isel-abort=1 -mtriple=x86_64-apple-darwin10 | FileCheck %s

define i32 @fcmp_oeq1(float %x) {
; CHECK-LABEL: fcmp_oeq1
; CHECK:       ucomiss  %xmm0, %xmm0
; CHECK-NEXT:  jp {{LBB.+_1}}
  %1 = fcmp oeq float %x, %x
  br i1 %1, label %bb1, label %bb2
bb2:
  ret i32 1
bb1:
  ret i32 0
}

define i32 @fcmp_oeq2(float %x) {
; CHECK-LABEL: fcmp_oeq2
; CHECK:       xorps    %xmm1, %xmm1
; CHECK-NEXT:  ucomiss  %xmm1, %xmm0
; CHECK-NEXT:  jne {{LBB.+_1}}
; CHECK-NEXT:  jp {{LBB.+_1}}
  %1 = fcmp oeq float %x, 0.000000e+00
  br i1 %1, label %bb1, label %bb2
bb2:
  ret i32 1
bb1:
  ret i32 0
}

define i32 @fcmp_ogt1(float %x) {
; CHECK-LABEL: fcmp_ogt1
; CHECK-NOT:   ucomiss
; CHECK:       movl $1, %eax
  %1 = fcmp ogt float %x, %x
  br i1 %1, label %bb1, label %bb2
bb2:
  ret i32 1
bb1:
  ret i32 0
}

define i32 @fcmp_ogt2(float %x) {
; CHECK-LABEL: fcmp_ogt2
; CHECK:       xorps    %xmm1, %xmm1
; CHECK-NEXT:  ucomiss  %xmm1, %xmm0
; CHECK-NEXT:  jbe {{LBB.+_1}}
  %1 = fcmp ogt float %x, 0.000000e+00
  br i1 %1, label %bb1, label %bb2
bb2:
  ret i32 1
bb1:
  ret i32 0
}

define i32 @fcmp_oge1(float %x) {
; CHECK-LABEL: fcmp_oge1
; CHECK:       ucomiss  %xmm0, %xmm0
; CHECK-NEXT:  jp {{LBB.+_1}}
  %1 = fcmp oge float %x, %x
  br i1 %1, label %bb1, label %bb2
bb2:
  ret i32 1
bb1:
  ret i32 0
}

define i32 @fcmp_oge2(float %x) {
; CHECK-LABEL: fcmp_oge2
; CHECK:       xorps    %xmm1, %xmm1
; CHECK-NEXT:  ucomiss  %xmm1, %xmm0
; CHECK-NEXT:  jb {{LBB.+_1}}
  %1 = fcmp oge float %x, 0.000000e+00
  br i1 %1, label %bb1, label %bb2
bb2:
  ret i32 1
bb1:
  ret i32 0
}

define i32 @fcmp_olt1(float %x) {
; CHECK-LABEL: fcmp_olt1
; CHECK-NOT:   ucomiss
; CHECK:       movl $1, %eax
  %1 = fcmp olt float %x, %x
  br i1 %1, label %bb1, label %bb2
bb2:
  ret i32 1
bb1:
  ret i32 0
}

define i32 @fcmp_olt2(float %x) {
; CHECK-LABEL: fcmp_olt2
; CHECK:       xorps    %xmm1, %xmm1
; CHECK-NEXT:  ucomiss  %xmm0, %xmm1
; CHECK-NEXT:  jbe {{LBB.+_1}}
  %1 = fcmp olt float %x, 0.000000e+00
  br i1 %1, label %bb1, label %bb2
bb2:
  ret i32 1
bb1:
  ret i32 0
}

define i32 @fcmp_ole1(float %x) {
; CHECK-LABEL: fcmp_ole1
; CHECK:       ucomiss  %xmm0, %xmm0
; CHECK-NEXT:  jp {{LBB.+_1}}
  %1 = fcmp ole float %x, %x
  br i1 %1, label %bb1, label %bb2
bb2:
  ret i32 1
bb1:
  ret i32 0
}

define i32 @fcmp_ole2(float %x) {
; CHECK-LABEL: fcmp_ole2
; CHECK:       xorps    %xmm1, %xmm1
; CHECK-NEXT:  ucomiss  %xmm0, %xmm1
; CHECK-NEXT:  jb {{LBB.+_1}}
  %1 = fcmp ole float %x, 0.000000e+00
  br i1 %1, label %bb1, label %bb2
bb2:
  ret i32 1
bb1:
  ret i32 0
}

define i32 @fcmp_one1(float %x) {
; CHECK-LABEL: fcmp_one1
; CHECK-NOT:   ucomiss
; CHECK:       movl $1, %eax
  %1 = fcmp one float %x, %x
  br i1 %1, label %bb1, label %bb2
bb2:
  ret i32 1
bb1:
  ret i32 0
}

define i32 @fcmp_one2(float %x) {
; CHECK-LABEL: fcmp_one2
; CHECK:       xorps    %xmm1, %xmm1
; CHECK-NEXT:  ucomiss  %xmm1, %xmm0
; CHECK-NEXT:  je {{LBB.+_1}}
  %1 = fcmp one float %x, 0.000000e+00
  br i1 %1, label %bb1, label %bb2
bb2:
  ret i32 1
bb1:
  ret i32 0
}

define i32 @fcmp_ord1(float %x) {
; CHECK-LABEL: fcmp_ord1
; CHECK:       ucomiss  %xmm0, %xmm0
; CHECK-NEXT:  jp {{LBB.+_1}}
  %1 = fcmp ord float %x, %x
  br i1 %1, label %bb1, label %bb2
bb2:
  ret i32 1
bb1:
  ret i32 0
}

define i32 @fcmp_ord2(float %x) {
; CHECK-LABEL: fcmp_ord2
; CHECK:       ucomiss  %xmm0, %xmm0
; CHECK-NEXT:  jp {{LBB.+_1}}
  %1 = fcmp ord float %x, 0.000000e+00
  br i1 %1, label %bb1, label %bb2
bb2:
  ret i32 1
bb1:
  ret i32 0
}

define i32 @fcmp_uno1(float %x) {
; CHECK-LABEL: fcmp_uno1
; CHECK:       ucomiss  %xmm0, %xmm0
; CHECK-NEXT:  jp {{LBB.+_2}}
  %1 = fcmp uno float %x, %x
  br i1 %1, label %bb1, label %bb2
bb2:
  ret i32 1
bb1:
  ret i32 0
}

define i32 @fcmp_uno2(float %x) {
; CHECK-LABEL: fcmp_uno2
; CHECK:       ucomiss  %xmm0, %xmm0
; CHECK-NEXT:  jp {{LBB.+_2}}
  %1 = fcmp uno float %x, 0.000000e+00
  br i1 %1, label %bb1, label %bb2
bb2:
  ret i32 1
bb1:
  ret i32 0
}

define i32 @fcmp_ueq1(float %x) {
; CHECK-LABEL: fcmp_ueq1
; CHECK-NOT:   ucomiss
  %1 = fcmp ueq float %x, %x
  br i1 %1, label %bb1, label %bb2
bb2:
  ret i32 1
bb1:
  ret i32 0
}

define i32 @fcmp_ueq2(float %x) {
; CHECK-LABEL: fcmp_ueq2
; CHECK:       xorps    %xmm1, %xmm1
; CHECK-NEXT:  ucomiss  %xmm1, %xmm0
; CHECK-NEXT:  je {{LBB.+_2}}
  %1 = fcmp ueq float %x, 0.000000e+00
  br i1 %1, label %bb1, label %bb2
bb2:
  ret i32 1
bb1:
  ret i32 0
}

define i32 @fcmp_ugt1(float %x) {
; CHECK-LABEL: fcmp_ugt1
; CHECK:       ucomiss  %xmm0, %xmm0
; CHECK-NEXT:  jnp {{LBB.+_1}}
  %1 = fcmp ugt float %x, %x
  br i1 %1, label %bb1, label %bb2
bb2:
  ret i32 1
bb1:
  ret i32 0
}

define i32 @fcmp_ugt2(float %x) {
; CHECK-LABEL: fcmp_ugt2
; CHECK:       xorps    %xmm1, %xmm1
; CHECK-NEXT:  ucomiss  %xmm0, %xmm1
; CHECK-NEXT:  jae {{LBB.+_1}}
  %1 = fcmp ugt float %x, 0.000000e+00
  br i1 %1, label %bb1, label %bb2
bb2:
  ret i32 1
bb1:
  ret i32 0
}

define i32 @fcmp_uge1(float %x) {
; CHECK-LABEL: fcmp_uge1
; CHECK-NOT:   ucomiss
  %1 = fcmp uge float %x, %x
  br i1 %1, label %bb1, label %bb2
bb2:
  ret i32 1
bb1:
  ret i32 0
}

define i32 @fcmp_uge2(float %x) {
; CHECK-LABEL: fcmp_uge2
; CHECK:       xorps    %xmm1, %xmm1
; CHECK-NEXT:  ucomiss  %xmm0, %xmm1
; CHECK-NEXT:  ja {{LBB.+_1}}
  %1 = fcmp uge float %x, 0.000000e+00
  br i1 %1, label %bb1, label %bb2
bb2:
  ret i32 1
bb1:
  ret i32 0
}

define i32 @fcmp_ult1(float %x) {
; CHECK-LABEL: fcmp_ult1
; CHECK:       ucomiss  %xmm0, %xmm0
; CHECK-NEXT:  jnp {{LBB.+_1}}
  %1 = fcmp ult float %x, %x
  br i1 %1, label %bb1, label %bb2
bb2:
  ret i32 1
bb1:
  ret i32 0
}

define i32 @fcmp_ult2(float %x) {
; CHECK-LABEL: fcmp_ult2
; CHECK:       xorps    %xmm1, %xmm1
; CHECK-NEXT:  ucomiss  %xmm1, %xmm0
; CHECK-NEXT:  jae {{LBB.+_1}}
  %1 = fcmp ult float %x, 0.000000e+00
  br i1 %1, label %bb1, label %bb2
bb2:
  ret i32 1
bb1:
  ret i32 0
}

define i32 @fcmp_ule1(float %x) {
; CHECK-LABEL: fcmp_ule1
; CHECK-NOT:   ucomiss
  %1 = fcmp ule float %x, %x
  br i1 %1, label %bb1, label %bb2
bb2:
  ret i32 1
bb1:
  ret i32 0
}

define i32 @fcmp_ule2(float %x) {
; CHECK-LABEL: fcmp_ule2
; CHECK:       xorps    %xmm1, %xmm1
; CHECK-NEXT:  ucomiss  %xmm1, %xmm0
; CHECK-NEXT:  ja {{LBB.+_1}}
  %1 = fcmp ule float %x, 0.000000e+00
  br i1 %1, label %bb1, label %bb2
bb2:
  ret i32 1
bb1:
  ret i32 0
}

define i32 @fcmp_une1(float %x) {
; CHECK-LABEL: fcmp_une1
; CHECK:       ucomiss  %xmm0, %xmm0
; CHECK-NEXT:  jnp {{LBB.+_1}}
  %1 = fcmp une float %x, %x
  br i1 %1, label %bb1, label %bb2
bb2:
  ret i32 1
bb1:
  ret i32 0
}

define i32 @fcmp_une2(float %x) {
; CHECK-LABEL: fcmp_une2
; CHECK:       xorps    %xmm1, %xmm1
; CHECK-NEXT:  ucomiss  %xmm1, %xmm0
; CHECK-NEXT:  jne {{LBB.+_2}}
; CHECK-NEXT:  jnp {{LBB.+_1}}
  %1 = fcmp une float %x, 0.000000e+00
  br i1 %1, label %bb1, label %bb2
bb2:
  ret i32 1
bb1:
  ret i32 0
}

define i32 @icmp_eq(i32 %x) {
; CHECK-LABEL: icmp_eq
; CHECK-NOT:   cmpl
; CHECK:       xorl %eax, %eax
  %1 = icmp eq i32 %x, %x
  br i1 %1, label %bb1, label %bb2
bb2:
  ret i32 1
bb1:
  ret i32 0
}

define i32 @icmp_ne(i32 %x) {
; CHECK-LABEL: icmp_ne
; CHECK-NOT:   cmpl
; CHECK:       movl $1, %eax
  %1 = icmp ne i32 %x, %x
  br i1 %1, label %bb1, label %bb2
bb2:
  ret i32 1
bb1:
  ret i32 0
}

define i32 @icmp_ugt(i32 %x) {
; CHECK-LABEL: icmp_ugt
; CHECK-NOT:   cmpl
; CHECK:       movl $1, %eax
  %1 = icmp ugt i32 %x, %x
  br i1 %1, label %bb1, label %bb2
bb2:
  ret i32 1
bb1:
  ret i32 0
}

define i32 @icmp_uge(i32 %x) {
; CHECK-LABEL: icmp_uge
; CHECK-NOT:   cmpl
; CHECK:       xorl %eax, %eax
  %1 = icmp uge i32 %x, %x
  br i1 %1, label %bb1, label %bb2
bb2:
  ret i32 1
bb1:
  ret i32 0
}

define i32 @icmp_ult(i32 %x) {
; CHECK-LABEL: icmp_ult
; CHECK-NOT:   cmpl
; CHECK:       movl $1, %eax
  %1 = icmp ult i32 %x, %x
  br i1 %1, label %bb1, label %bb2
bb2:
  ret i32 1
bb1:
  ret i32 0
}

define i32 @icmp_ule(i32 %x) {
; CHECK-LABEL: icmp_ule
; CHECK-NOT:   cmpl
; CHECK:       xorl %eax, %eax
  %1 = icmp ule i32 %x, %x
  br i1 %1, label %bb1, label %bb2
bb2:
  ret i32 1
bb1:
  ret i32 0
}

define i32 @icmp_sgt(i32 %x) {
; CHECK-LABEL: icmp_sgt
; CHECK-NOT:   cmpl
; CHECK:       movl $1, %eax
  %1 = icmp sgt i32 %x, %x
  br i1 %1, label %bb1, label %bb2
bb2:
  ret i32 1
bb1:
  ret i32 0
}

define i32 @icmp_sge(i32 %x) {
; CHECK-LABEL: icmp_sge
; CHECK-NOT:   cmpl
; CHECK:       xorl %eax, %eax
  %1 = icmp sge i32 %x, %x
  br i1 %1, label %bb1, label %bb2
bb2:
  ret i32 1
bb1:
  ret i32 0
}

define i32 @icmp_slt(i32 %x) {
; CHECK-LABEL: icmp_slt
; CHECK-NOT:   cmpl
; CHECK:       movl $1, %eax
  %1 = icmp slt i32 %x, %x
  br i1 %1, label %bb1, label %bb2
bb2:
  ret i32 1
bb1:
  ret i32 0
}

define i32 @icmp_sle(i32 %x) {
; CHECK-LABEL: icmp_sle
; CHECK-NOT:   cmpl
; CHECK:       xorl %eax, %eax
  %1 = icmp sle i32 %x, %x
  br i1 %1, label %bb1, label %bb2
bb2:
  ret i32 1
bb1:
  ret i32 0
}

