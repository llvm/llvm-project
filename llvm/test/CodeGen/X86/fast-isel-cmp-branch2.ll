; RUN: llc < %s                             -mtriple=x86_64-apple-darwin10 | FileCheck %s
; RUN: llc < %s -fast-isel -fast-isel-abort=1 -mtriple=x86_64-apple-darwin10 | FileCheck %s

define i32 @fcmp_oeq(float %x, float %y) {
; CHECK-LABEL: fcmp_oeq
; CHECK:       ucomiss  %xmm1, %xmm0
; CHECK-NEXT:  jne {{LBB.+_1}}
; CHECK-NEXT:  jnp {{LBB.+_2}}
  %1 = fcmp oeq float %x, %y
  br i1 %1, label %bb1, label %bb2
bb2:
  ret i32 1
bb1:
  ret i32 0
}

define i32 @fcmp_ogt(float %x, float %y) {
; CHECK-LABEL: fcmp_ogt
; CHECK:       ucomiss  %xmm1, %xmm0
; CHECK-NEXT:  jbe {{LBB.+_1}}
  %1 = fcmp ogt float %x, %y
  br i1 %1, label %bb1, label %bb2
bb2:
  ret i32 1
bb1:
  ret i32 0
}

define i32 @fcmp_oge(float %x, float %y) {
; CHECK-LABEL: fcmp_oge
; CHECK:       ucomiss  %xmm1, %xmm0
; CHECK-NEXT:  jb {{LBB.+_1}}
  %1 = fcmp oge float %x, %y
  br i1 %1, label %bb1, label %bb2
bb2:
  ret i32 1
bb1:
  ret i32 0
}

define i32 @fcmp_olt(float %x, float %y) {
; CHECK-LABEL: fcmp_olt
; CHECK:       ucomiss  %xmm0, %xmm1
; CHECK-NEXT:  jbe {{LBB.+_1}}
  %1 = fcmp olt float %x, %y
  br i1 %1, label %bb1, label %bb2
bb2:
  ret i32 1
bb1:
  ret i32 0
}

define i32 @fcmp_ole(float %x, float %y) {
; CHECK-LABEL: fcmp_ole
; CHECK:       ucomiss  %xmm0, %xmm1
; CHECK-NEXT:  jb {{LBB.+_1}}
  %1 = fcmp ole float %x, %y
  br i1 %1, label %bb1, label %bb2
bb2:
  ret i32 1
bb1:
  ret i32 0
}

define i32 @fcmp_one(float %x, float %y) {
; CHECK-LABEL: fcmp_one
; CHECK:       ucomiss  %xmm1, %xmm0
; CHECK-NEXT:  je {{LBB.+_1}}
  %1 = fcmp one float %x, %y
  br i1 %1, label %bb1, label %bb2
bb2:
  ret i32 1
bb1:
  ret i32 0
}

define i32 @fcmp_ord(float %x, float %y) {
; CHECK-LABEL: fcmp_ord
; CHECK:       ucomiss  %xmm1, %xmm0
; CHECK-NEXT:  jp {{LBB.+_1}}
  %1 = fcmp ord float %x, %y
  br i1 %1, label %bb1, label %bb2
bb2:
  ret i32 1
bb1:
  ret i32 0
}

define i32 @fcmp_uno(float %x, float %y) {
; CHECK-LABEL: fcmp_uno
; CHECK:       ucomiss  %xmm1, %xmm0
; CHECK-NEXT:  jp {{LBB.+_2}}
  %1 = fcmp uno float %x, %y
  br i1 %1, label %bb1, label %bb2
bb2:
  ret i32 1
bb1:
  ret i32 0
}

define i32 @fcmp_ueq(float %x, float %y) {
; CHECK-LABEL: fcmp_ueq
; CHECK:       ucomiss  %xmm1, %xmm0
; CHECK-NEXT:  je {{LBB.+_2}}
  %1 = fcmp ueq float %x, %y
  br i1 %1, label %bb1, label %bb2
bb2:
  ret i32 1
bb1:
  ret i32 0
}

define i32 @fcmp_ugt(float %x, float %y) {
; CHECK-LABEL: fcmp_ugt
; CHECK:       ucomiss  %xmm0, %xmm1
; CHECK-NEXT:  jae {{LBB.+_1}}
  %1 = fcmp ugt float %x, %y
  br i1 %1, label %bb1, label %bb2
bb2:
  ret i32 1
bb1:
  ret i32 0
}

define i32 @fcmp_uge(float %x, float %y) {
; CHECK-LABEL: fcmp_uge
; CHECK:       ucomiss  %xmm0, %xmm1
; CHECK-NEXT:  ja {{LBB.+_1}}
  %1 = fcmp uge float %x, %y
  br i1 %1, label %bb1, label %bb2
bb2:
  ret i32 1
bb1:
  ret i32 0
}

define i32 @fcmp_ult(float %x, float %y) {
; CHECK-LABEL: fcmp_ult
; CHECK:       ucomiss  %xmm1, %xmm0
; CHECK-NEXT:  jae {{LBB.+_1}}
  %1 = fcmp ult float %x, %y
  br i1 %1, label %bb1, label %bb2
bb2:
  ret i32 1
bb1:
  ret i32 0
}

define i32 @fcmp_ule(float %x, float %y) {
; CHECK-LABEL: fcmp_ule
; CHECK:       ucomiss  %xmm1, %xmm0
; CHECK-NEXT:  ja {{LBB.+_1}}
  %1 = fcmp ule float %x, %y
  br i1 %1, label %bb1, label %bb2
bb2:
  ret i32 1
bb1:
  ret i32 0
}

define i32 @fcmp_une(float %x, float %y) {
; CHECK-LABEL: fcmp_une
; CHECK:       ucomiss  %xmm1, %xmm0
; CHECK-NEXT:  jne {{LBB.+_2}}
; CHECK-NEXT:  jp  {{LBB.+_2}}
; CHECK-NEXT:  jmp {{LBB.+_1}}
  %1 = fcmp une float %x, %y
  br i1 %1, label %bb1, label %bb2
bb2:
  ret i32 1
bb1:
  ret i32 0
}

define i32 @icmp_eq(i32 %x, i32 %y) {
; CHECK-LABEL: icmp_eq
; CHECK:       cmpl %esi, %edi
; CHECK-NEXT:  jne {{LBB.+_1}}
  %1 = icmp eq i32 %x, %y
  br i1 %1, label %bb1, label %bb2
bb2:
  ret i32 1
bb1:
  ret i32 0
}

define i32 @icmp_ne(i32 %x, i32 %y) {
; CHECK-LABEL: icmp_ne
; CHECK:       cmpl %esi, %edi
; CHECK-NEXT:  je {{LBB.+_1}}
  %1 = icmp ne i32 %x, %y
  br i1 %1, label %bb1, label %bb2
bb2:
  ret i32 1
bb1:
  ret i32 0
}

define i32 @icmp_ugt(i32 %x, i32 %y) {
; CHECK-LABEL: icmp_ugt
; CHECK:       cmpl %esi, %edi
; CHECK-NEXT:  jbe {{LBB.+_1}}
  %1 = icmp ugt i32 %x, %y
  br i1 %1, label %bb1, label %bb2
bb2:
  ret i32 1
bb1:
  ret i32 0
}

define i32 @icmp_uge(i32 %x, i32 %y) {
; CHECK-LABEL: icmp_uge
; CHECK:       cmpl %esi, %edi
; CHECK-NEXT:  jb {{LBB.+_1}}
  %1 = icmp uge i32 %x, %y
  br i1 %1, label %bb1, label %bb2
bb2:
  ret i32 1
bb1:
  ret i32 0
}

define i32 @icmp_ult(i32 %x, i32 %y) {
; CHECK-LABEL: icmp_ult
; CHECK:       cmpl %esi, %edi
; CHECK-NEXT:  jae {{LBB.+_1}}
  %1 = icmp ult i32 %x, %y
  br i1 %1, label %bb1, label %bb2
bb2:
  ret i32 1
bb1:
  ret i32 0
}

define i32 @icmp_ule(i32 %x, i32 %y) {
; CHECK-LABEL: icmp_ule
; CHECK:       cmpl %esi, %edi
; CHECK-NEXT:  ja {{LBB.+_1}}
  %1 = icmp ule i32 %x, %y
  br i1 %1, label %bb1, label %bb2
bb2:
  ret i32 1
bb1:
  ret i32 0
}

define i32 @icmp_sgt(i32 %x, i32 %y) {
; CHECK-LABEL: icmp_sgt
; CHECK:       cmpl %esi, %edi
; CHECK-NEXT:  jle {{LBB.+_1}}
  %1 = icmp sgt i32 %x, %y
  br i1 %1, label %bb1, label %bb2
bb2:
  ret i32 1
bb1:
  ret i32 0
}

define i32 @icmp_sge(i32 %x, i32 %y) {
; CHECK-LABEL: icmp_sge
; CHECK:       cmpl %esi, %edi
; CHECK-NEXT:  jl {{LBB.+_1}}
  %1 = icmp sge i32 %x, %y
  br i1 %1, label %bb1, label %bb2
bb2:
  ret i32 1
bb1:
  ret i32 0
}

define i32 @icmp_slt(i32 %x, i32 %y) {
; CHECK-LABEL: icmp_slt
; CHECK:       cmpl %esi, %edi
; CHECK-NEXT:  jge {{LBB.+_1}}
  %1 = icmp slt i32 %x, %y
  br i1 %1, label %bb1, label %bb2
bb2:
  ret i32 1
bb1:
  ret i32 0
}

define i32 @icmp_sle(i32 %x, i32 %y) {
; CHECK-LABEL: icmp_sle
; CHECK:       cmpl %esi, %edi
; CHECK-NEXT:  jg {{LBB.+_1}}
  %1 = icmp sle i32 %x, %y
  br i1 %1, label %bb1, label %bb2
bb2:
  ret i32 1
bb1:
  ret i32 0
}

