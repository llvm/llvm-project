; RUN: %python <%s -c 'import re,sys; sys.stdout.write(re.sub(r" (nuw|nsw|exact|disjoint|nneg|samesign|inbounds|nusw|nnan)( |$)", r"\2", sys.stdin.read(), flags=re.M))' > %t.ll
; RUN: llvm-diff %s %s
; RUN: not llvm-diff %s %t.ll 2>&1 | FileCheck %s --implicit-check-not=no_flags

; CHECK-LABEL: in function add_nuw:
; CHECK: {{^ *<}} {{.*}}add nuw i32
define i32 @add_nuw(i32 %x) {
  %r = add nuw i32 %x, 1
  ret i32 %r
}

; CHECK-LABEL: in function add_nsw:
; CHECK: {{^ *<}} {{.*}}add nsw i32
define i32 @add_nsw(i32 %x) {
  %r = add nsw i32 %x, 1
  ret i32 %r
}

; CHECK-LABEL: in function sub_nuw:
; CHECK: {{^ *<}} {{.*}}sub nuw i32
define i32 @sub_nuw(i32 %x) {
  %r = sub nuw i32 %x, 1
  ret i32 %r
}

; CHECK-LABEL: in function mul_nsw:
; CHECK: {{^ *<}} {{.*}}mul nsw i32
define i32 @mul_nsw(i32 %x) {
  %r = mul nsw i32 %x, 3
  ret i32 %r
}

; CHECK-LABEL: in function shl_nuw:
; CHECK: {{^ *<}} {{.*}}shl nuw i32
define i32 @shl_nuw(i32 %x) {
  %r = shl nuw i32 %x, 2
  ret i32 %r
}

; CHECK-LABEL: in function trunc_nsw:
; CHECK: {{^ *<}} {{.*}}trunc nsw i32
define i16 @trunc_nsw(i32 %x) {
  %r = trunc nsw i32 %x to i16
  ret i16 %r
}

; CHECK-LABEL: in function udiv_exact:
; CHECK: {{^ *<}} {{.*}}udiv exact i32
define i32 @udiv_exact(i32 %x) {
  %r = udiv exact i32 %x, 3
  ret i32 %r
}

; CHECK-LABEL: in function lshr_exact:
; CHECK: {{^ *<}} {{.*}}lshr exact i32
define i32 @lshr_exact(i32 %x) {
  %r = lshr exact i32 %x, 1
  ret i32 %r
}

; CHECK-LABEL: in function or_disjoint:
; CHECK: {{^ *<}} {{.*}}or disjoint i32
define i32 @or_disjoint(i32 %x, i32 %y) {
  %r = or disjoint i32 %x, %y
  ret i32 %r
}

; CHECK-LABEL: in function zext_nneg:
; CHECK: {{^ *<}} {{.*}}zext nneg i32
define i64 @zext_nneg(i32 %x) {
  %r = zext nneg i32 %x to i64
  ret i64 %r
}

; CHECK-LABEL: in function uitofp_nneg:
; CHECK: {{^ *<}} {{.*}}uitofp nneg i32
define float @uitofp_nneg(i32 %x) {
  %r = uitofp nneg i32 %x to float
  ret float %r
}

; CHECK-LABEL: in function icmp_samesign:
; CHECK: {{^ *<}} {{.*}}icmp samesign ult
define i1 @icmp_samesign(i32 %x, i32 %y) {
  %r = icmp samesign ult i32 %x, %y
  ret i1 %r
}

; CHECK-LABEL: in function gep_inbounds:
; CHECK: {{^ *<}} {{.*}}getelementptr inbounds i32
define ptr @gep_inbounds(ptr %p, i64 %i) {
  %r = getelementptr inbounds i32, ptr %p, i64 %i
  ret ptr %r
}

; CHECK-LABEL: in function gep_inbounds_nuw:
; CHECK: {{^ *<}} {{.*}}getelementptr inbounds nuw i32
define ptr @gep_inbounds_nuw(ptr %p, i64 %i) {
  %r = getelementptr inbounds nuw i32, ptr %p, i64 %i
  ret ptr %r
}

; CHECK-LABEL: in function gep_nusw:
; CHECK: {{^ *<}} {{.*}}getelementptr nusw i32
define ptr @gep_nusw(ptr %p, i64 %i) {
  %r = getelementptr nusw i32, ptr %p, i64 %i
  ret ptr %r
}

; CHECK-LABEL: in function fadd_nnan:
; CHECK: {{^ *<}} {{.*}}fadd nnan float
define float @fadd_nnan(float %x, float %y) {
  %r = fadd nnan float %x, %y
  ret float %r
}

define i32 @no_flags(i32 %x) {
  %r = add i32 %x, 1
  ret i32 %r
}
