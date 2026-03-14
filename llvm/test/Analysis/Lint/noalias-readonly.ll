; RUN: opt < %s -passes=lint -disable-output 2>&1 | FileCheck %s

declare void @f1(ptr noalias readonly, ptr)

define void @f2(ptr %a) {
entry:
  call void @f1(ptr %a, ptr %a)
  ret void
}

; Lint should complain about us passing %a to both arguments, since the noalias
; argument may depend on writes to the other.
; CHECK: Unusual: noalias argument aliases another argument
; CHECK-NEXT: call void @f1(ptr %a, ptr %a)

declare void @f3(ptr noalias, ptr readonly)

define void @f4(ptr %a) {
entry:
  call void @f3(ptr %a, ptr %a)
  ret void
}

; Lint should complain about us passing %a to both arguments, since writes to
; the noalias argument may cause a dependency for the other.
; CHECK: Unusual: noalias argument aliases another argument
; CHECK-NEXT: call void @f3(ptr %a, ptr %a)

declare void @f5(ptr noalias readonly, ptr readonly)

define void @f6(ptr %a) {
entry:
  call void @f5(ptr %a, ptr %a)
  ret void
}

; Lint should not complain about passing %a to both arguments even if one is
; noalias, since they are both readonly and thus have no dependence.
; CHECK-NOT: Unusual: noalias argument aliases another argument
; CHECK-NOT: call void @f5(ptr %a, ptr %a)
