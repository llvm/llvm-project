; RUN: opt < %s -verify-ipgo -debug-only=verify-ipgo -passes=instcombine -S -disable-output 2>&1 | FileCheck %s --check-prefix=DEFAULT
; RUN: opt < %s -verify-ipgo -debug-only=verify-ipgo -passes=instcombine -pass-remarks-analysis=verify-ipgo -S -disable-output 2>&1 | FileCheck %s --check-prefix=DEFAULT-REMARK
; RUN: opt < %s -verify-ipgo -passes=instcombine -S -disable-output 2>&1 | FileCheck %s --check-prefix=VERIFY
; REQUIRES: asserts
;
; Verify default global-function skip behavior in caller-site sum checks:
; - By default, globally visible functions are skipped.
; - Current branch emits unknown-block-frequency diagnostics for main in this case,
;   but does not report entry-count mismatch for the global callee.

define i32 @callee(i32 %x) !prof !10 {
entry:
  %y = add i32 %x, 1
  ret i32 %y
}

define i32 @main() {
entry:
  %c = call i32 @callee(i32 0)
  br label %exit

exit:
  %v = add i32 %c, 0
  ret i32 %v
}

; DEFAULT-NOT: PGOVerify# Entry count mismatch in function callee
; DEFAULT: PGOVerify# Not able to determine Block frequency for main, block entry

; DEFAULT-REMARK-NOT: remark: <unknown>:0:0: Entry count mismatch: entry=2 vs caller-sum=1

; VERIFY: *** IPGO Verification After InstCombinePass ***
; VERIFY-NOT: PGOVerify# Entry count mismatch in function callee

!llvm.module.flags = !{!30}
!30 = !{i32 1, !"ProfileSummary", !31}
!31 = !{!32, !33, !34, !35, !36, !37, !38, !39}
!32 = !{!"ProfileFormat", !"InstrProf"}
!33 = !{!"TotalCount", i64 3}
!34 = !{!"MaxCount", i64 2}
!35 = !{!"MaxInternalCount", i64 2}
!36 = !{!"MaxFunctionCount", i64 2}
!37 = !{!"NumCounts", i64 2}
!38 = !{!"NumFunctions", i64 2}
!39 = !{!"DetailedSummary", !40}
!40 = !{!41, !42, !43}
!41 = !{i32 10000, i64 2, i32 1}
!42 = !{i32 999000, i64 2, i32 1}
!43 = !{i32 999999, i64 1, i32 2}

!10 = !{!"function_entry_count", i64 2}
!20 = !{!"branch_weights", i32 1}
