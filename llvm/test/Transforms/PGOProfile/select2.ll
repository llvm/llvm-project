; RUN: llvm-profdata merge %S/Inputs/select2.proftext -o %t.profdata
; RUN: opt < %s -pgo-instr-use -pgo-test-profile-file=%t.profdata -pgo-instr-select=true -S | FileCheck %s --check-prefix=USE
; RUN: opt < %s -passes=pgo-instr-use -pgo-test-profile-file=%t.profdata -pgo-instr-select=true -S | FileCheck %s --check-prefix=USE

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define i32 @foo(i32 %n) {
;USE: define i32 @foo(i32 %n) !prof ![[ENTRY_COUNT:[0-9]+]] {
entry:
  br label %for.cond

for.cond:
  %i.0 = phi i32 [ 0, %entry ], [ %inc, %for.inc ]
  %sum.0 = phi i32 [ 0, %entry ], [ %add, %for.inc ]
  %cmp = icmp slt i32 %i.0, %n
  br i1 %cmp, label %for.body, label %for.end
;USE:  br i1 %cmp, label %for.body, label %for.end, !prof ![[BW_FOR_BR:[0-9]+]]

for.body:
  %cmp1 = icmp sgt i32 %sum.0, 10
  %cond = select i1 %cmp1, i32 20, i32 -10
;USE:  %cond = select i1 %cmp1, i32 20, i32 -10, !prof ![[BW_FOR_SELECT:[0-9]+]]
  %add = add nsw i32 %sum.0, %cond
  br label %for.inc

for.inc:
  %inc = add nsw i32 %i.0, 1
  br label %for.cond

for.end:
  ret i32 %sum.0
}

;USE: ![[ENTRY_COUNT]] = !{!"function_entry_count", i64 3}
;USE: ![[BW_FOR_BR]] = !{!"branch_weights", i32 800, i32 3}
;USE: ![[BW_FOR_SELECT]] = !{!"branch_weights", i32 300, i32 500}
