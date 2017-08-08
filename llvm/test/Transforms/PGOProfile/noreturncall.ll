; RUN: llvm-profdata merge %S/Inputs/noreturncall.proftext -o %t.profdata
; RUN: opt < %s -pgo-instr-use -pgo-test-profile-file=%t.profdata -S | FileCheck %s --check-prefix=USE
; RUN: opt < %s -passes=pgo-instr-use -pgo-test-profile-file=%t.profdata -S | FileCheck %s --check-prefix=USE
; REQUIRES: asserts

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare i32 @bar0(i32)

define i32 @bar2(i32 %i) {
entry:
  unreachable
}

define i32 @foo(i32 %i, i32 %j, i32 %k) {
entry:
  %cmp = icmp slt i32 %i, 999
  br i1 %cmp, label %if.then, label %if.end
; USE: br i1 %cmp, label %if.then, label %if.end
; USE-SAME: !prof ![[BW_ENTRY:[0-9]+]]

if.then:
  %call = call i32 @bar0(i32 %i)
  br label %if.end

if.end:
  %ret.0 = phi i32 [ %call, %if.then ], [ 0, %entry ]
  %cmp1 = icmp sgt i32 %j, 1000
  %cmp3 = icmp sgt i32 %k, 99
  %or.cond = and i1 %cmp1, %cmp3
  br i1 %or.cond, label %if.then4, label %if.end7
; USE: br i1 %or.cond, label %if.then4, label %if.end7
; USE-SAME: !prof ![[BW_IF:[0-9]+]]

if.then4:
  %call5 = call i32 @bar2(i32 undef)
  br label %if.end7

if.end7:
  %mul = mul nsw i32 %ret.0, %ret.0
  ret i32 %mul
}
; USE: ![[BW_ENTRY]] = !{!"branch_weights", i32 21, i32 0}
; USE: ![[BW_IF]] = !{!"branch_weights", i32 0, i32 20}
