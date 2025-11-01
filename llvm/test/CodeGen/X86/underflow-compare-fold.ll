; RUN: llc < %s -mtriple=x86_64 | FileCheck %s

; GitHub issue #161036

; Positive test : umin(sub(a,b),a) with scalar types should be folded
define i64 @underflow_compare_fold(i64 %a, i64 %b) {
; CHECK-LABEL: underflow_compare_fold
; CHECK-LABEL: %bb.0
; CHECK-NEXT: movq    %rdi, %rax
; CHECK-NEXT: subq    %rsi, %rax
; CHECK-NEXT: cmovbq  %rdi, %rax
; CHECK-NEXT: retq
  %sub = sub i64 %a, %b
  %cond = tail call i64 @llvm.umin.i64(i64 %sub, i64 %a)
  ret i64 %cond
}

; Negative test, vector types : umin(sub(a,b),a) but with vectors
define <16 x i8> @underflow_compare_dontfold_vectors(<16 x i8> %a, <16 x i8> %b) {
; CHECK-LABEL: underflow_compare_dontfold_vectors
; CHECK-LABEL: %bb.0
; CHECK-NEXT: movdqa %xmm0, %xmm2
; CHECK-NEXT: psubb %xmm1, %xmm2
; CHECK-NEXT: pminub %xmm2, %xmm0
; CHECK-NEXT: retq
  %sub = sub <16 x i8> %a, %b
  %cond = tail call <16 x i8> @llvm.umin.v16i8(<16 x i8> %sub, <16 x i8> %a)
  ret <16 x i8> %cond
}

; Negative test, pattern mismatch : umin(a,sub(a,b))
define i64 @umin_sub_inverse_args(i64 %a, i64 %b) {
; CHECK-LABEL: umin_sub_inverse_args
; CHECK-LABEL: %bb.0
; CHECK-NEXT: movq %rdi, %rax
; CHECK-NEXT: subq %rsi, %rax
; CHECK-NEXT: cmpq %rax, %rdi
; CHECK-NEXT: cmovbq %rdi, %rax
; CHECK-NEXT: retq
  %sub = sub i64 %a, %b
  %cond = tail call i64 @llvm.umin.i64(i64 %a, i64 %sub)
  ret i64 %cond
}

; Negative test, pattern mismatch : umin(add(a,b),a)
define i64 @umin_add(i64 %a, i64 %b) {
; CHECK-LABEL: umin_add
; CHECK-LABEL: %bb.0
; CHECK-NEXT: leaq (%rsi,%rdi), %rax
; CHECK-NEXT: cmpq %rdi, %rax
; CHECK-NEXT: cmovaeq %rdi, %rax
; CHECK-NEXT: retq
  %add = add i64 %a, %b
  %cond = tail call i64 @llvm.umin.i64(i64 %add, i64 %a)
  ret i64 %cond
}
