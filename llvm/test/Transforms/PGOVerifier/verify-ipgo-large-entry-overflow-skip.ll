; REQUIRES: asserts
; RUN: opt < %s -verify-ipgo -debug-only=verify-ipgo -passes=instcombine -S -disable-output 2>&1 | FileCheck %s
; RUN: opt < %s -verify-ipgo -passes=instcombine -S -disable-output 2>&1 | FileCheck %s --check-prefix=VERIFY
;
; Mother-patch overflow gating test (trimmed to block-frequency diagnostics).
; - overflow_loop has entry count > uint32 max and should be skipped.
; - normal_loop has small entry count and should still emit unknown-frequency
;   diagnostics in this synthetic setup.

define internal i32 @overflow_loop(i32 %n) !prof !10 {
entry:
  %x = add i32 %n, 0
  br label %header

header:
  %i = phi i32 [ 0, %entry ], [ %inc, %latch ]
  %cmp = icmp slt i32 %i, %x
  br i1 %cmp, label %body, label %exit

body:
  br label %latch

latch:
  %inc = add i32 %i, 1
  br label %header

exit:
  ret i32 %i
}

define internal i32 @normal_loop(i32 %n) !prof !11 {
entry:
  %x = add i32 %n, 0
  br label %header

header:
  %i = phi i32 [ 0, %entry ], [ %inc, %latch ]
  %cmp = icmp slt i32 %i, %x
  br i1 %cmp, label %body, label %exit

body:
  br label %latch

latch:
  %inc = add i32 %i, 1
  br label %header

exit:
  ret i32 %i
}

define i32 @main() {
entry:
  %a = call i32 @overflow_loop(i32 4)
  %b = call i32 @normal_loop(i32 4)
  %s = add i32 %a, %b
  ret i32 %s
}

; CHECK-NOT: PGOVerify# Not able to determine Block frequency for overflow_loop
; CHECK: PGOVerify# Not able to determine Block frequency for normal_loop, block header

; VERIFY: *** IPGO Verification After InstCombinePass ***

!llvm.module.flags = !{!30}
!30 = !{i32 1, !"ProfileSummary", !31}
!31 = !{!32, !33, !34, !35, !36, !37, !38, !39}
!32 = !{!"ProfileFormat", !"InstrProf"}
!33 = !{!"TotalCount", i64 4294967297}
!34 = !{!"MaxCount", i64 4294967296}
!35 = !{!"MaxInternalCount", i64 4294967296}
!36 = !{!"MaxFunctionCount", i64 4294967296}
!37 = !{!"NumCounts", i64 2}
!38 = !{!"NumFunctions", i64 2}
!39 = !{!"DetailedSummary", !40}
!40 = !{!41, !42, !43}
!41 = !{i32 10000, i64 4294967296, i32 1}
!42 = !{i32 999000, i64 4294967296, i32 1}
!43 = !{i32 999999, i64 1, i32 2}

!10 = !{!"function_entry_count", i64 4294967296}
!11 = !{!"function_entry_count", i64 1}
