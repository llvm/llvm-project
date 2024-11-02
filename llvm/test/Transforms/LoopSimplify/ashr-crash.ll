; RUN: opt -passes='loop-mssa(loop-rotate,licm),instcombine,indvars,loop-unroll' -S %s | FileCheck %s
;
; PR18361: ScalarEvolution::getAddRecExpr():
;          Assertion `isLoopInvariant(Operands[i],...
;
; After a series of loop optimizations, SCEV's LoopDispositions grow stale.
; In particular, LoopSimplify hoists %cmp4, resulting in this SCEV for %add:
; {(zext i1 %cmp4 to i32),+,1}<nw><%for.cond1.preheader>
;
; When recomputing the SCEV for %ashr, we truncate the operands to get:
; (zext i1 %cmp4 to i16)
;
; This SCEV was never mapped to a value so never invalidated. It's
; loop disposition is still marked as non-loop-invariant, which is
; inconsistent with the AddRec.

target datalayout = "e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx"

@d = common global i32 0, align 4
@a = common global i32 0, align 4
@c = common global i32 0, align 4
@b = common global i32 0, align 4

; Check that the def-use chain that leads to the bad SCEV is still
; there.
;
; CHECK-LABEL: @foo
; CHECK-LABEL: entry:
; CHECK-LABEL: for.cond1.preheader:
; CHECK-LABEL: for.body3:
; CHECK: %cmp4.le.le
; CHECK: %conv.le.le = zext i1 %cmp4.le.le to i32
; CHECK: %xor.le.le = xor i32 %conv6.le.le, 1
define void @foo() {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc7, %entry
  %storemerge = phi i32 [ 0, %entry ], [ %inc8, %for.inc7 ]
  %f.0 = phi i32 [ undef, %entry ], [ %f.1, %for.inc7 ]
  store i32 %storemerge, i32* @d, align 4
  %cmp = icmp slt i32 %storemerge, 1
  br i1 %cmp, label %for.cond1, label %for.end9

for.cond1:                                        ; preds = %for.cond, %for.body3
  %storemerge1 = phi i32 [ %inc, %for.body3 ], [ 0, %for.cond ]
  %f.1 = phi i32 [ %xor, %for.body3 ], [ %f.0, %for.cond ]
  store i32 %storemerge1, i32* @a, align 4
  %cmp2 = icmp slt i32 %storemerge1, 1
  br i1 %cmp2, label %for.body3, label %for.inc7

for.body3:                                        ; preds = %for.cond1
  %0 = load i32, i32* @c, align 4
  %cmp4 = icmp sge i32 %storemerge1, %0
  %conv = zext i1 %cmp4 to i32
  %1 = load i32, i32* @d, align 4
  %add = add nsw i32 %conv, %1
  %sext = shl i32 %add, 16
  %conv6 = ashr exact i32 %sext, 16
  %xor = xor i32 %conv6, 1
  %inc = add nsw i32 %storemerge1, 1
  br label %for.cond1

for.inc7:                                         ; preds = %for.cond1
  %2 = load i32, i32* @d, align 4
  %inc8 = add nsw i32 %2, 1
  br label %for.cond

for.end9:                                         ; preds = %for.cond
  %cmp10 = icmp sgt i32 %f.0, 0
  br i1 %cmp10, label %if.then, label %if.end

if.then:                                          ; preds = %for.end9
  store i32 0, i32* @b, align 4
  br label %if.end

if.end:                                           ; preds = %if.then, %for.end9
  ret void
}
