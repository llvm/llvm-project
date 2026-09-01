; REQUIRES: asserts
; RUN: opt -debug-only=verify-ipgo -verify-ipgo -passes='instcombine' -disable-output %s 2>&1 | FileCheck %s
; RUN: opt -verify-ipgo -passes='instcombine' -disable-output %s 2>&1 | FileCheck %s --check-prefix=VERIFY
;
; Mother-patch-derived block-frequency test, intentionally checking only
; block-frequency mismatch diagnostics.

define i32 @inconsistent_entry(i32 %x) !prof !0 {
entry:
  %cmp = icmp sgt i32 %x, 0
  br i1 %cmp, label %positive, label %negative, !prof !1

positive:
  %mul = mul nsw i32 %x, 2
  %sink1 = add nsw i32 %mul, 0
  ret i32 %sink1

negative:
  %div = sdiv i32 %x, 2
  %sink2 = add nsw i32 %div, 0
  ret i32 %sink2
}

define i32 @inconsistent_loop(i32 %n) !prof !2 {
entry:
  br label %loop.header

loop.header:
  %i = phi i32 [ 0, %entry ], [ %inc, %loop.body ]
  %sum = phi i32 [ 0, %entry ], [ %add, %loop.body ]
  %cmp = icmp slt i32 %i, %n
  br i1 %cmp, label %loop.body, label %loop.exit, !prof !3

loop.body:
  %add = add nsw i32 %sum, %i
  %inc = add nsw i32 %i, 1
  br label %loop.header

loop.exit:
  ret i32 %sum
}

define i32 @inconsistent_branches(i32 %a, i32 %b) !prof !5 {
entry:
  %cmp1 = icmp sgt i32 %a, 0
  br i1 %cmp1, label %then1, label %else1, !prof !6

then1:
  %mul = mul nsw i32 %a, 2
  br label %middle

else1:
  %div = sdiv i32 %a, 2
  br label %middle

middle:
  %val = phi i32 [ %mul, %then1 ], [ %div, %else1 ]
  %cmp2 = icmp sgt i32 %b, 0
  br i1 %cmp2, label %then2, label %else2, !prof !9

then2:
  %add = add nsw i32 %val, %b
  br label %end

else2:
  %sub = sub nsw i32 %val, %b
  br label %end

end:
  %result = phi i32 [ %add, %then2 ], [ %sub, %else2 ]
  ret i32 %result
}

define i32 @inconsistent_switch(i32 %x) !prof !12 {
entry:
  switch i32 %x, label %default [
    i32 1, label %case1
    i32 2, label %case2
    i32 3, label %case3
  ], !prof !13

case1:
  ret i32 10

case2:
  ret i32 20

case3:
  ret i32 30

default:
  ret i32 0
}

; CHECK: *** IPGO Verification After InstCombinePass ***
; CHECK: PGOVerify cache invalidated
; CHECK: PGOVerify# Block frequency mismatch in function inconsistent_entry, block entry: Incoming=1000: Outgoing=900
; CHECK: PGOVerify# Block frequency mismatch in function inconsistent_branches, block middle: Incoming=1000: Outgoing=900
; CHECK: PGOVerify# Block frequency mismatch in function inconsistent_switch, block entry: Incoming=1000: Outgoing=900

; VERIFY: *** IPGO Verification After InstCombinePass ***
; VERIFY: PGOVerify# Block frequency mismatch in function inconsistent_entry, block entry: Incoming=1000: Outgoing=900
; VERIFY: PGOVerify# Block frequency mismatch in function inconsistent_branches, block middle: Incoming=1000: Outgoing=900
; VERIFY: PGOVerify# Block frequency mismatch in function inconsistent_switch, block entry: Incoming=1000: Outgoing=900

!0 = !{!"function_entry_count", i64 1000}
!1 = !{!"branch_weights", i32 700, i32 200}
!2 = !{!"function_entry_count", i64 100}
!3 = !{!"branch_weights", i32 900, i32 100}
!4 = !{!"branch_weights", i32 800}
!5 = !{!"function_entry_count", i64 1000}
!6 = !{!"branch_weights", i32 600, i32 400}
!7 = !{!"branch_weights", i32 600}
!8 = !{!"branch_weights", i32 400}
!9 = !{!"branch_weights", i32 700, i32 200}
!10 = !{!"branch_weights", i32 700}
!11 = !{!"branch_weights", i32 200}
!12 = !{!"function_entry_count", i64 1000}
!13 = !{!"branch_weights", i32 100, i32 200, i32 300, i32 300}

!llvm.module.flags = !{!20}
!20 = !{i32 1, !"ProfileSummary", !21}
!21 = !{!22, !23, !24, !25, !26, !27, !28, !29}
!22 = !{!"ProfileFormat", !"InstrProf"}
!23 = !{!"TotalCount", i64 4000}
!24 = !{!"MaxCount", i64 1000}
!25 = !{!"MaxInternalCount", i64 1000}
!26 = !{!"MaxFunctionCount", i64 1000}
!27 = !{!"NumCounts", i64 13}
!28 = !{!"NumFunctions", i64 4}
!29 = !{!"DetailedSummary", !30}
!30 = !{!31, !32, !33}
!31 = !{i32 10000, i64 1000, i32 1}
!32 = !{i32 999000, i64 900, i32 4}
!33 = !{i32 999999, i64 100, i32 13}
