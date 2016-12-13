; RUN: llvm-profdata merge %S/Inputs/noreturncall.proftext -o %t.profdata
; RUN: opt < %s -pgo-instr-use -pgo-test-profile-file=%t.profdata -S -debug-only=pgo-instrumentation 2>&1 | FileCheck %s --check-prefix=USE
; RUN: opt < %s -passes=pgo-instr-use -pgo-test-profile-file=%t.profdata -S -debug-only=pgo-instrumentation 2>&1 | FileCheck %s --check-prefix=USE
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

if.then:
  %call = call i32 @bar0(i32 %i)
  br label %if.end

if.end:
  %ret.0 = phi i32 [ %call, %if.then ], [ 0, %entry ]
  %cmp1 = icmp sgt i32 %j, 1000
  %cmp3 = icmp sgt i32 %k, 99
  %or.cond = and i1 %cmp1, %cmp3
  br i1 %or.cond, label %if.then4, label %if.end7

if.then4:
  %call5 = call i32 @bar2(i32 undef)
  br label %if.end7

if.end7:
  %mul = mul nsw i32 %ret.0, %ret.0
  ret i32 %mul
}
; USE:  Edge 0: 1-->3  c  W=8000  Count=0
; USE:  Edge 1: 3-->5  c  W=8000  Count=20
; USE:  Edge 2: 0-->1     W=16  Count=21
; USE:  Edge 3: 5-->0 *   W=16  Count=20
; USE:  Edge 4: 1-->2     W=8  Count=21
; USE:  Edge 5: 2-->3 *   W=8  Count=21
; USE:  Edge 6: 3-->4     W=8  Count=0
; USE:  Edge 7: 4-->5 *   W=8  Count=0
