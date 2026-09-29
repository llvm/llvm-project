; RUN: llc < %s -mtriple=arm64-eabi -enable-misched=false -verify-machineinstrs | FileCheck %s

; The next set of tests makes sure we can combine the second instruction into
; the first.

; CHECK-LABEL: ldp_int_aa
; CHECK: ldp w8, w9, [x1]
; CHECK: str w0, [x1, #8]
; CHECK: ret
define i32 @ldp_int_aa(i32 %a, ptr %p) nounwind {
  %tmp = load i32, ptr %p, align 4
  %str.ptr = getelementptr inbounds i32, ptr %p, i64 2
  store i32 %a, ptr %str.ptr, align 4
  %add.ptr = getelementptr inbounds i32, ptr %p, i64 1
  %tmp1 = load i32, ptr %add.ptr, align 4
  %add = add nsw i32 %tmp1, %tmp
  ret i32 %add
}

; CHECK-LABEL: ldp_long_aa
; CHECK: ldp x8, x9, [x1]
; CHECK: str x0, [x1, #16]
; CHECK: ret
define i64 @ldp_long_aa(i64 %a, ptr %p) nounwind {
  %tmp = load i64, ptr %p, align 8
  %str.ptr = getelementptr inbounds i64, ptr %p, i64 2
  store i64 %a, ptr %str.ptr, align 4
  %add.ptr = getelementptr inbounds i64, ptr %p, i64 1
  %tmp1 = load i64, ptr %add.ptr, align 8
  %add = add nsw i64 %tmp1, %tmp
  ret i64 %add
}

; CHECK-LABEL: ldp_float_aa
; CHECK: str s0, [x0, #8]
; CHECK: ldp s1, s0, [x0]
; CHECK: ret
define float @ldp_float_aa(float %a, ptr %p) nounwind {
  %tmp = load float, ptr %p, align 4
  %str.ptr = getelementptr inbounds float, ptr %p, i64 2
  store float %a, ptr %str.ptr, align 4
  %add.ptr = getelementptr inbounds float, ptr %p, i64 1
  %tmp1 = load float, ptr %add.ptr, align 4
  %add = fadd float %tmp, %tmp1
  ret float %add
}

; CHECK-LABEL: ldp_double_aa
; CHECK: str d0, [x0, #16]
; CHECK: ldp d1, d0, [x0]
; CHECK: ret
define double @ldp_double_aa(double %a, ptr %p) nounwind {
  %tmp = load double, ptr %p, align 8
  %str.ptr = getelementptr inbounds double, ptr %p, i64 2
  store double %a, ptr %str.ptr, align 4
  %add.ptr = getelementptr inbounds double, ptr %p, i64 1
  %tmp1 = load double, ptr %add.ptr, align 8
  %add = fadd double %tmp, %tmp1
  ret double %add
}
