; RUN: opt -S -passes=hotcoldsplit -hotcoldsplit-threshold=-1 < %s 2>&1 | FileCheck %s

; CHECK-LABEL: define {{.*}} @foo(
; CHECK-NOT: llvm.assume
; CHECK: call void @foo.cold.1()
; CHECK: llvm.assume
; CHECK-NEXT: ret void

; CHECK-LABEL: define {{.*}} @foo.cold.1(
; CHECK-NOT: llvm.assume
; CHECK: call void @cold()
; CHECK-NOT: llvm.assume
; CHECK: }

define void @foo(i1 %cond) {
entry:
  br i1 %cond, label %cold, label %cont

cold:
  call void @llvm.assume(i1 %cond)
  call void @cold()
  br label %cont

cont:
  %cmp = icmp eq i1 %cond, true
  br i1 %cmp, label %exit1, label %exit2

exit1:
  call void @llvm.assume(i1 %cond)
  ret void

exit2:
  ret void
}

declare void @llvm.assume(i1)

declare void @cold() cold
