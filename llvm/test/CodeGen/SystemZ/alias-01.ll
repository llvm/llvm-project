; Test 32-bit ANDs in which the second operand is variable.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

; Check that there are no spills.
define void @f1(<16 x i32> *%src1, <16 x float> *%dest) {
; CHECK-LABEL: f1:
; CHECK-NOT: %r15
; CHECK: br %r14
  %val = load <16 x i32>, <16 x i32> *%src1, !tbaa !1
  %add = add <16 x i32> %val, %val
  %res = bitcast <16 x i32> %add to <16 x float>
  store <16 x float> %res, <16 x float> *%dest, !tbaa !2
  ret void
}

!0 = !{ !"root" }
!1 = !{ !"set1", !0 }
!2 = !{ !"set2", !0 }
