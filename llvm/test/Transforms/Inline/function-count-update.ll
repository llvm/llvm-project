; RUN: opt < %s -passes='require<profile-summary>,cgscc(inline)' -S | FileCheck %s

; This tests that the function count of two callees get correctly updated after
; they have been inlined into two back-to-back callsites in a single basic block
; in the caller. The callees have the alwaysinline attribute and so they get
; inlined both with the regular inliner pass and the always inline pass. In
; both cases, the new count of each callee is the original count minus callsite
; count which is 200 (since the caller's entry count is 400 and the block
; containing the calls have a relative block frequency of 0.5).

; CHECK: @callee1(i32 %n) #0 !prof [[COUNT1:![0-9]+]]
define i32 @callee1(i32 %n) #0 !prof !1 {
  %cond = icmp sle i32 %n, 10
  br i1 %cond, label %cond_true, label %cond_false

cond_true:
  %r1 = add i32 %n, 1
  ret i32 %r1
cond_false:
  %r2 = add i32 %n, 2
  ret i32 %r2
}

; CHECK: @callee2(i32 %n) #0 !prof [[COUNT2:![0-9]+]]
define i32 @callee2(i32 %n) #0 !prof !2 {
  %r1 = add i32 %n, 1
  ret i32 %r1
}

define i32 @caller(i32 %n) !prof !3 {
  %cond = icmp sle i32 %n, 100
  br i1 %cond, label %cond_true, label %cond_false

cond_true:
  %i = call i32 @callee1(i32 %n)
  %j = call i32 @callee2(i32 %i)
  ret i32 %j
cond_false:
  ret i32 0
}

!llvm.module.flags = !{!0}
; CHECK: [[COUNT1]] = !{!"function_entry_count", i64 800}
; CHECK: [[COUNT2]] = !{!"function_entry_count", i64 1800}
!0 = !{i32 1, !"MaxFunctionCount", i32 1000}
!1 = !{!"function_entry_count", i64 1000}
!2 = !{!"function_entry_count", i64 2000}
!3 = !{!"function_entry_count", i64 400}
attributes #0 = { alwaysinline }

