; RUN: llc < %s -mtriple=x86_64-apple-macosx -pre-RA-sched=source | FileCheck %s

; Teach two-address pass to update the "source" map so it doesn't perform a
; non-profitable commute using outdated info. The test case would still fail
; because of poor pre-RA schedule. That will be fixed by MI scheduler.
; rdar://11472010
define i32 @t(i32 %mask) nounwind readnone ssp {
entry:
; CHECK-LABEL: t:
; CHECK-NOT: mov
  %sub = add i32 %mask, -65535
  %shr = lshr i32 %sub, 23
  %and = and i32 %mask, 1
  %add = add i32 %shr, %and
  ret i32 %add
}
