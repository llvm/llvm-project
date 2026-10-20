; RUN: llc -mtriple=x86_64-unknown-linux-gnu -mcpu=haswell < %s | FileCheck %s

; AND is more likely to evaluate both operands when the true edge is hot;
; OR is more likely to evaluate both operands when the false edge is hot.

declare void @yes()
declare void @no()

; CHECK-LABEL: and_cold:
; CHECK: cmpl
; CHECK: jne no@PLT
; CHECK: cmpl
; CHECK: jne no@PLT
; CHECK: jmp yes@PLT
define void @and_cold(i32 %a, i32 %b, i32 %c, i32 %d) {
entry:
  %lhs = icmp eq i32 %a, %b
  %rhs = icmp eq i32 %c, %d
  %cond = and i1 %lhs, %rhs
  br i1 %cond, label %true, label %false, !prof !0
true:
  tail call void @yes()
  ret void
false:
  tail call void @no()
  ret void
}

; CHECK-LABEL: and_hot:
; CHECK: xorl
; CHECK: xorl
; CHECK: orl
; CHECK: jne no@PLT
; CHECK: jmp yes@PLT
define void @and_hot(i32 %a, i32 %b, i32 %c, i32 %d) {
entry:
  %lhs = icmp eq i32 %a, %b
  %rhs = icmp eq i32 %c, %d
  %cond = and i1 %lhs, %rhs
  br i1 %cond, label %true, label %false, !prof !1
true:
  tail call void @yes()
  ret void
false:
  tail call void @no()
  ret void
}

; CHECK-LABEL: or_hot:
; CHECK: cmpl
; CHECK: jne yes@PLT
; CHECK: cmpl
; CHECK: je no@PLT
; CHECK: jmp yes@PLT
define void @or_hot(i32 %a, i32 %b, i32 %c, i32 %d) {
entry:
  %lhs = icmp ne i32 %a, %b
  %rhs = icmp ne i32 %c, %d
  %cond = or i1 %lhs, %rhs
  br i1 %cond, label %true, label %false, !prof !1
true:
  tail call void @yes()
  ret void
false:
  tail call void @no()
  ret void
}

; CHECK-LABEL: or_cold:
; CHECK: xorl
; CHECK: xorl
; CHECK: orl
; CHECK: je no@PLT
; CHECK: jmp yes@PLT
define void @or_cold(i32 %a, i32 %b, i32 %c, i32 %d) {
entry:
  %lhs = icmp ne i32 %a, %b
  %rhs = icmp ne i32 %c, %d
  %cond = or i1 %lhs, %rhs
  br i1 %cond, label %true, label %false, !prof !0
true:
  tail call void @yes()
  ret void
false:
  tail call void @no()
  ret void
}

!0 = !{!"branch_weights", i32 1, i32 2000}
!1 = !{!"branch_weights", i32 2000, i32 1}
