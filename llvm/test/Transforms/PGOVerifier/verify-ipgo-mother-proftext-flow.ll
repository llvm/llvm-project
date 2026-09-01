; RUN: llvm-profdata merge %S/Inputs/verify-ipgo-mother-block-frequency.proftext -o %t.profdata
; RUN: opt < %s -verify-ipgo -verify-ipgo-print-diagnostics -passes=pgo-instr-use -pgo-test-profile-file=%t.profdata -S -disable-output 2>&1 | FileCheck %s
;
; Mother-patch proftext pipeline coverage:
; profile text -> profdata -> pgo-instr-use -> verify-ipgo.
;
; CHECK: *** IPGO Verification After PGOInstrumentationUse ***
; CHECK: PGOVerify# Block frequency mismatch in function inconsistent_entry, block entry: Incoming=1000: Outgoing=900
; CHECK: PGOVerify# Block frequency mismatch in function inconsistent_branches, block middle: Incoming=1000: Outgoing=900
; CHECK: PGOVerify# Block frequency mismatch in function inconsistent_switch, block entry: Incoming=1000: Outgoing=900

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define i32 @consistent_flow(i32 noundef %n) !prof !0 {
entry:
  %cmp = icmp sgt i32 %n, 10
  br i1 %cmp, label %if.then, label %if.else, !prof !1

if.then:
  %add = add nsw i32 %n, 5
  br label %if.end

if.else:
  %sub = sub nsw i32 %n, 3
  br label %if.end

if.end:
  %result = phi i32 [ %add, %if.then ], [ %sub, %if.else ]
  ret i32 %result
}

define i32 @inconsistent_entry(i32 noundef %x) !prof !4 {
entry:
  %cmp = icmp sgt i32 %x, 0
  br i1 %cmp, label %positive, label %negative, !prof !5

positive:
  %mul = mul nsw i32 %x, 2
  ret i32 %mul

negative:
  %div = sdiv i32 %x, 2
  ret i32 %div
}

define i32 @inconsistent_loop(i32 noundef %n) !prof !6 {
entry:
  br label %loop.header

loop.header:
  %i = phi i32 [ 0, %entry ], [ %inc, %loop.body ]
  %sum = phi i32 [ 0, %entry ], [ %add, %loop.body ]
  %cmp = icmp slt i32 %i, %n
  br i1 %cmp, label %loop.body, label %loop.exit, !prof !7

loop.body:
  %add = add nsw i32 %sum, %i
  %inc = add nsw i32 %i, 1
  br label %loop.header

loop.exit:
  ret i32 %sum
}

define i32 @inconsistent_branches(i32 noundef %a, i32 noundef %b) !prof !9 {
entry:
  %cmp1 = icmp sgt i32 %a, 0
  br i1 %cmp1, label %then1, label %else1, !prof !10

then1:
  %mul = mul nsw i32 %a, 2
  br label %middle

else1:
  %div = sdiv i32 %a, 2
  br label %middle

middle:
  %val = phi i32 [ %mul, %then1 ], [ %div, %else1 ]
  %cmp2 = icmp sgt i32 %b, 0
  br i1 %cmp2, label %then2, label %else2, !prof !13

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

define i32 @inconsistent_switch(i32 noundef %x) !prof !16 {
entry:
  switch i32 %x, label %default [
    i32 1, label %case1
    i32 2, label %case2
    i32 3, label %case3
  ], !prof !17

case1:
  ret i32 10

case2:
  ret i32 20

case3:
  ret i32 30

default:
  ret i32 0
}

!0 = !{!"function_entry_count", i64 1000}
!1 = !{!"branch_weights", i32 600, i32 400}
!2 = !{!"branch_weights", i32 600}
!3 = !{!"branch_weights", i32 400}
!4 = !{!"function_entry_count", i64 1000}
!5 = !{!"branch_weights", i32 700, i32 200}
!6 = !{!"function_entry_count", i64 100}
!7 = !{!"branch_weights", i32 900, i32 100}
!8 = !{!"branch_weights", i32 800}
!9 = !{!"function_entry_count", i64 1000}
!10 = !{!"branch_weights", i32 600, i32 400}
!11 = !{!"branch_weights", i32 600}
!12 = !{!"branch_weights", i32 400}
!13 = !{!"branch_weights", i32 700, i32 200}
!14 = !{!"branch_weights", i32 700}
!15 = !{!"branch_weights", i32 200}
!16 = !{!"function_entry_count", i64 1000}
!17 = !{!"branch_weights", i32 100, i32 200, i32 300, i32 300}
