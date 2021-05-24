; RUN: opt -basic-aa -sroa -loop-rotate -licm -S < %s | FileCheck %s
; RUN: opt -basic-aa -sroa -loop-rotate %s | opt -aa-pipeline=basic-aa -passes='require<aa>,require<targetir>,require<scalar-evolution>,require<opt-remark-emit>,loop(licm)' -S | FileCheck %s
; RUN: opt -basic-aa -sroa -loop-rotate -licm -enable-mssa-loop-dependency=true -verify-memoryssa -S < %s | FileCheck %s
; The objects *p and *q are aliased to each other, but even though *q is
; volatile, *p can be considered invariant in the loop. Check if it is moved
; out of the loop.
; CHECK: load i32, i32* %p
; CHECK: for.body:
; CHECK: load volatile i32, i32* %q

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

; Function Attrs: nounwind uwtable
define i32 @foo(i32* dereferenceable(4) nonnull %p, i32* %q, i32 %n) #0 {
entry:
  %p.addr = alloca i32*, align 8
  %q.addr = alloca i32*, align 8
  %n.addr = alloca i32, align 4
  %i = alloca i32, align 4
  %s = alloca i32, align 4
  store i32* %p, i32** %p.addr, align 8
  store i32* %q, i32** %q.addr, align 8
  store i32 %n, i32* %n.addr, align 4
  store i32 0, i32* %s, align 4
  store i32 0, i32* %i, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %0 = load i32, i32* %i, align 4
  %1 = load i32, i32* %n.addr, align 4
  %cmp = icmp slt i32 %0, %1
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %2 = load i32*, i32** %p.addr, align 8
  %3 = load i32, i32* %2, align 4
  %4 = load i32*, i32** %q.addr, align 8
  %5 = load volatile i32, i32* %4, align 4
  %add = add nsw i32 %3, %5
  %6 = load i32, i32* %s, align 4
  %add1 = add nsw i32 %6, %add
  store i32 %add1, i32* %s, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %7 = load i32, i32* %i, align 4
  %inc = add nsw i32 %7, 1
  store i32 %inc, i32* %i, align 4
  br label %for.cond

for.end:                                          ; preds = %for.cond
  %8 = load i32, i32* %s, align 4
  ret i32 %8
}

attributes #0 = { nounwind uwtable "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
