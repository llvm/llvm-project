; REQUIRES: asserts
; RUN: opt -debug-only=verify-ipgo -verify-ipgo -passes='instcombine' -disable-output %s 2>&1 | FileCheck %s
; RUN: opt -verify-ipgo -passes='instcombine' -disable-output %s 2>&1 | FileCheck %s --check-prefix=VERIFY
;
; flow-conservation test, intentionally checking only
; block-frequency mismatch diagnostics.

define i32 @incorrect_if_else_middle(i32 %x) !prof !0 {
entry:
  %cmp = icmp sgt i32 %x, 0
  br i1 %cmp, label %if.then, label %if.else, !prof !1

if.then:
  %add = add i32 %x, 5
  %cmp2 = icmp sgt i32 %add, 0
  br i1 %cmp2, label %merge, label %if.then.cont, !prof !2

if.then.cont:
  br label %merge

if.else:
  %sub = sub i32 %x, 5
  br label %merge

merge:
  %val = phi i32 [ %add, %if.then ], [ %add, %if.then.cont ], [ %sub, %if.else ]
  %mul = mul i32 %val, 2
  ret i32 %mul
}

; CHECK: *** IPGO Verification After InstCombinePass ***
; CHECK: PGOVerify cache invalidated
; CHECK: PGOVerify# Block frequency mismatch in function incorrect_if_else_middle, block if.then: Incoming=700: Outgoing=600

; VERIFY: *** IPGO Verification After InstCombinePass ***
; VERIFY: PGOVerify# Block frequency mismatch in function incorrect_if_else_middle, block if.then: Incoming=700: Outgoing=600

!0 = !{!"function_entry_count", i64 1000}
!1 = !{!"branch_weights", i32 700, i32 300}
!2 = !{!"branch_weights", i32 400, i32 200}

!llvm.module.flags = !{!10}
!10 = !{i32 1, !"ProfileSummary", !11}
!11 = !{!12, !13, !14, !15, !16, !17, !18, !19}
!12 = !{!"ProfileFormat", !"InstrProf"}
!13 = !{!"TotalCount", i64 1000}
!14 = !{!"MaxCount", i64 700}
!15 = !{!"MaxInternalCount", i64 700}
!16 = !{!"MaxFunctionCount", i64 1000}
!17 = !{!"NumCounts", i64 3}
!18 = !{!"NumFunctions", i64 1}
!19 = !{!"DetailedSummary", !20}
!20 = !{!21, !22, !23}
!21 = !{i32 10000, i64 700, i32 1}
!22 = !{i32 999000, i64 500, i32 2}
!23 = !{i32 999999, i64 300, i32 3}
