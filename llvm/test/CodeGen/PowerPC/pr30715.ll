; RUN: llc -verify-machineinstrs -mcpu=pwr8 -mtriple=powerpc64le-unknown-linux-gnu < %s | FileCheck %s

%class.FullMatrix = type { i8 }
%class.Vector = type { ptr }

$test = comdat any

define weak_odr void @test(ptr %this, ptr dereferenceable(8) %p1, ptr dereferenceable(8), i1 zeroext) {
entry:
  %call = tail call signext i32 @fn1(ptr %this)
  %cmp10 = icmp sgt i32 %call, 0
  br i1 %cmp10, label %for.body.lr.ph, label %for.cond.cleanup

for.body.lr.ph:                                   ; preds = %entry
  %2 = load ptr, ptr %p1, align 8
  %wide.trip.count = zext i32 %call to i64
  %min.iters.check = icmp ult i32 %call, 4
  br i1 %min.iters.check, label %for.body.preheader, label %min.iters.checked

for.body.preheader:                               ; preds = %middle.block, %min.iters.checked, %for.body.lr.ph
  %indvars.iv.ph = phi i64 [ 0, %min.iters.checked ], [ 0, %for.body.lr.ph ], [ %n.vec, %middle.block ]
  br label %for.body

min.iters.checked:                                ; preds = %for.body.lr.ph
  %3 = and i32 %call, 3
  %n.mod.vf = zext i32 %3 to i64
  %n.vec = sub nsw i64 %wide.trip.count, %n.mod.vf
  %cmp.zero = icmp eq i64 %n.vec, 0
  br i1 %cmp.zero, label %for.body.preheader, label %vector.body.preheader

vector.body.preheader:                            ; preds = %min.iters.checked
  br label %vector.body

vector.body:                                      ; preds = %vector.body.preheader, %vector.body
  %index = phi i64 [ %index.next, %vector.body ], [ 0, %vector.body.preheader ]
  %4 = getelementptr inbounds float, ptr %2, i64 %index
  %wide.load = load <4 x float>, ptr %4, align 4
  %5 = fpext <4 x float> %wide.load to <4 x ppc_fp128>
  %6 = fadd <4 x ppc_fp128> %5, %5
  %7 = fptrunc <4 x ppc_fp128> %6 to <4 x float>
  store <4 x float> %7, ptr %4, align 4
  %index.next = add i64 %index, 4
  %8 = icmp eq i64 %index.next, %n.vec
  br i1 %8, label %middle.block, label %vector.body

middle.block:                                     ; preds = %vector.body
  %cmp.n = icmp eq i32 %3, 0
  br i1 %cmp.n, label %for.cond.cleanup, label %for.body.preheader

for.cond.cleanup.loopexit:                        ; preds = %for.body
  br label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.cond.cleanup.loopexit, %middle.block, %entry
  ret void

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ %indvars.iv.ph, %for.body.preheader ]
  %arrayidx.i = getelementptr inbounds float, ptr %2, i64 %indvars.iv
  %9 = load float, ptr %arrayidx.i, align 4
  %conv = fpext float %9 to ppc_fp128
  %add = fadd ppc_fp128 %conv, %conv
  %conv4 = fptrunc ppc_fp128 %add to float
  store float %conv4, ptr %arrayidx.i, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond, label %for.cond.cleanup.loopexit, label %for.body
; CHECK: stfd
; CHECK: lxvd2x
}

declare signext i32 @fn1(ptr) local_unnamed_addr #1
