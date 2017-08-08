; RUN: llc -o - %s | FileCheck %s
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx"

; selectiondag stack protector uses a GuardReg which the fast-isel stack
; protection code did not but the state was not reset properly.
; The optnone attribute on @bar forces fast-isel.

; CHECK-LABEL: foo:
; CHECK: movq ___stack_chk_guard@GOTPCREL(%rip), %rax
; CHECK: movq ___stack_chk_guard@GOTPCREL(%rip), %rax
define void @foo() #0 {
entry:
  %_tags = alloca [3 x i32], align 4
  ret void
}

; CHECK-LABEL: bar:
; CHECK: movq ___stack_chk_guard@GOTPCREL(%rip), %{{r.x}}
; CHECK-DAG: movq ___stack_chk_guard@GOTPCREL(%rip), %[[GUARD:r.x]]
; CHECK-DAG: movq {{[0-9]+}}(%rsp), %[[CANARY:r.x]]
; CHECK: subq %[[CANARY]], %[[GUARD]]
define void @bar() #1 {
entry:
  %vt = alloca [2 x double], align 16
  br i1 undef, label %cleanup.4091, label %for.cond.3850

unreachable:
  unreachable

for.cond.3850:
  br i1 undef, label %land.rhs.3853, label %land.end.3857

land.rhs.3853:
  br label %land.end.3857

land.end.3857:
  %0 = phi i1 [ false, %for.cond.3850 ], [ false, %land.rhs.3853 ]
  br i1 %0, label %unreachable, label %unreachable

cleanup.4091:
  ret void
}

attributes #0 = { ssp }
attributes #1 = { noinline optnone ssp }
