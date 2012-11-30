; RUN: opt -indvars -S < %s | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"

@X = external global [0 x double]

; Indvars should be able to simplify simple comparisons involving
; induction variables.

; CHECK: @foo
; CHECK: %cond = and i1 %tobool.not, true

define void @foo(i64 %n, i32* nocapture %p) nounwind {
entry:
  %cmp9 = icmp sgt i64 %n, 0
  br i1 %cmp9, label %pre, label %return

pre:
  %t3 = load i32* %p
  %tobool.not = icmp ne i32 %t3, 0
  br label %loop

loop:
  %i = phi i64 [ 0, %pre ], [ %inc, %for.inc ]
  %cmp6 = icmp slt i64 %i, %n
  %cond = and i1 %tobool.not, %cmp6
  br i1 %cond, label %if.then, label %for.inc

if.then:
  %arrayidx = getelementptr [0 x double]* @X, i64 0, i64 %i
  store double 3.200000e+00, double* %arrayidx
  br label %for.inc

for.inc:
  %inc = add nsw i64 %i, 1
  %exitcond = icmp sge i64 %inc, %n
  br i1 %exitcond, label %return, label %loop

return:
  ret void
}

; Don't eliminate an icmp that's contributing to the loop exit test though.

; CHECK: @_ZNK4llvm5APInt3ultERKS0_
; CHECK: %tmp99 = icmp sgt i32 %i, -1

define i32 @_ZNK4llvm5APInt3ultERKS0_(i32 %tmp2.i1, i64** %tmp65, i64** %tmp73, i64** %tmp82, i64** %tmp90) {
entry:
  br label %bb18

bb13:
  %tmp66 = load i64** %tmp65, align 4
  %tmp68 = getelementptr inbounds i64* %tmp66, i32 %i
  %tmp69 = load i64* %tmp68, align 4
  %tmp74 = load i64** %tmp73, align 4
  %tmp76 = getelementptr inbounds i64* %tmp74, i32 %i
  %tmp77 = load i64* %tmp76, align 4
  %tmp78 = icmp ugt i64 %tmp69, %tmp77
  br i1 %tmp78, label %bb20.loopexit, label %bb15

bb15:
  %tmp83 = load i64** %tmp82, align 4
  %tmp85 = getelementptr inbounds i64* %tmp83, i32 %i
  %tmp86 = load i64* %tmp85, align 4
  %tmp91 = load i64** %tmp90, align 4
  %tmp93 = getelementptr inbounds i64* %tmp91, i32 %i
  %tmp94 = load i64* %tmp93, align 4
  %tmp95 = icmp ult i64 %tmp86, %tmp94
  br i1 %tmp95, label %bb20.loopexit, label %bb17

bb17:
  %tmp97 = add nsw i32 %i, -1
  br label %bb18

bb18:
  %i = phi i32 [ %tmp2.i1, %entry ], [ %tmp97, %bb17 ]
  %tmp99 = icmp sgt i32 %i, -1
  br i1 %tmp99, label %bb13, label %bb20.loopexit

bb20.loopexit:
  %tmp.0.ph = phi i32 [ 0, %bb18 ], [ 1, %bb15 ], [ 0, %bb13 ]
  ret i32 %tmp.0.ph
}

; Indvars should eliminate the icmp here.

; CHECK: @func_10
; CHECK-NOT: icmp
; CHECK: ret void

define void @func_10() nounwind {
entry:
  br label %loop

loop:
  %i = phi i32 [ %i.next, %loop ], [ 0, %entry ]
  %t0 = icmp slt i32 %i, 0
  %t1 = zext i1 %t0 to i32
  %t2 = add i32 %t1, %i
  %u3 = zext i32 %t2 to i64
  store i64 %u3, i64* null
  %i.next = add i32 %i, 1
  br i1 undef, label %loop, label %return

return:
  ret void
}

; PR14432
; Indvars should not turn the second loop into an infinite one.

; CHECK: @func_11
; CHECK: %tmp5 = icmp slt i32 %__key6.0, 10
; CHECK-NOT: br i1 true, label %noassert68, label %unrolledend

define i32 @func_11() nounwind uwtable {
entry:
  br label %forcond

forcond:                                          ; preds = %noassert, %entry
  %__key6.0 = phi i32 [ 2, %entry ], [ %tmp37, %noassert ]
  %tmp5 = icmp slt i32 %__key6.0, 10
  br i1 %tmp5, label %noassert, label %forcond38.preheader

forcond38.preheader:                              ; preds = %forcond
  br label %forcond38

noassert:                                         ; preds = %forbody
  %tmp13 = sdiv i32 -32768, %__key6.0
  %tmp2936 = shl i32 %tmp13, 24
  %sext23 = shl i32 %tmp13, 24
  %tmp32 = icmp eq i32 %tmp2936, %sext23
  %tmp37 = add i32 %__key6.0, 1
  br i1 %tmp32, label %forcond, label %assert33

assert33:                                         ; preds = %noassert
  tail call void @llvm.trap()
  unreachable

forcond38:                                        ; preds = %noassert68, %forcond38.preheader
  %__key8.0 = phi i32 [ %tmp81, %noassert68 ], [ 2, %forcond38.preheader ]
  %tmp46 = icmp slt i32 %__key8.0, 10
  br i1 %tmp46, label %noassert68, label %unrolledend

noassert68:                                       ; preds = %forbody39
  %tmp57 = sdiv i32 -32768, %__key8.0
  %sext34 = shl i32 %tmp57, 16
  %sext21 = shl i32 %tmp57, 16
  %tmp76 = icmp eq i32 %sext34, %sext21
  %tmp81 = add i32 %__key8.0, 1
  br i1 %tmp76, label %forcond38, label %assert77

assert77:                                         ; preds = %noassert68
  tail call void @llvm.trap()
  unreachable

unrolledend:                                      ; preds = %forcond38
  ret i32 0
}

declare void @llvm.trap() noreturn nounwind
