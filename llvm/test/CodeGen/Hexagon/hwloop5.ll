; RUN: llc -O3 -mtriple=hexagon -mcpu=hexagonv5 < %s | FileCheck %s
;
; Generate hardware loop when unknown trip count loop is vectorized.

; CHECK: loop0(.LBB{{[0-9]*}}_{{[0-9]*}},r{{[0-9]+}})
; CHECK: endloop0
; CHECK: loop0(.LBB{{[0-9]*}}_{{[0-9]*}},r{{[0-9]+}})
; CHECK: endloop0

@A = common global [1000 x i32] zeroinitializer, align 8
@B = common global [1000 x i32] zeroinitializer, align 8

define i32 @dotprod2(i32 %count) #0 {
entry.split:
  %cmp6 = icmp sgt i32 %count, 0
  br i1 %cmp6, label %polly.cond, label %for.end

for.end.loopexit:
  br label %for.end

for.end:
  %sum.0.lcssa.reg2mem.0.load37 = phi i32 [ 0, %entry.split ], [ %p_add34, %polly.loop_if13 ], [ %p_add, %for.end.loopexit ]
  ret i32 %sum.0.lcssa.reg2mem.0.load37

polly.cond:
  %0 = icmp sgt i32 %count, 1
  br i1 %0, label %polly.loop_if, label %polly.loop_if13

polly.loop_exit.loopexit:
  br label %polly.loop_exit

polly.loop_exit:
  %1 = phi <2 x i32> [ zeroinitializer, %polly.loop_if ], [ %addp_vec, %polly.loop_exit.loopexit ]
  %2 = extractelement <2 x i32> %1, i32 0
  %3 = extractelement <2 x i32> %1, i32 1
  %add_sum = add i32 %2, %3
  br label %polly.loop_if13

polly.loop_if:
  %4 = add i32 %count, -1
  %leftover_lb = and i32 %4, -2
  %polly.loop_guard = icmp eq i32 %leftover_lb, 0
  br i1 %polly.loop_guard, label %polly.loop_exit, label %polly.loop_preheader

polly.stmt.for.body:
  %addp_vec28 = phi <2 x i32> [ zeroinitializer, %polly.loop_preheader ], [ %addp_vec, %polly.stmt.for.body ]
  %scevgep.phi = phi ptr [ @A, %polly.loop_preheader ], [ %scevgep.inc, %polly.stmt.for.body ]
  %scevgep9.phi = phi ptr [ @B, %polly.loop_preheader ], [ %scevgep9.inc, %polly.stmt.for.body ]
  %polly.indvar = phi i32 [ 0, %polly.loop_preheader ], [ %polly.indvar_next, %polly.stmt.for.body ]
  %_p_vec_full = load <2 x i32>, ptr %scevgep.phi, align 8
  %_p_vec_full11 = load <2 x i32>, ptr %scevgep9.phi, align 8
  %mulp_vec = mul <2 x i32> %_p_vec_full11, %_p_vec_full
  %addp_vec = add <2 x i32> %mulp_vec, %addp_vec28
  %polly.indvar_next = add nsw i32 %polly.indvar, 2
  %polly.loop_cond = icmp eq i32 %polly.indvar, %polly.adjust_ub
  %scevgep.inc = getelementptr i32, ptr %scevgep.phi, i32 2
  %scevgep9.inc = getelementptr i32, ptr %scevgep9.phi, i32 2
  br i1 %polly.loop_cond, label %polly.loop_exit.loopexit, label %polly.stmt.for.body

polly.loop_preheader:
  %polly.adjust_ub = add i32 %leftover_lb, -2
  br label %polly.stmt.for.body

polly.loop_if13:
  %p_add34 = phi i32 [ 0, %polly.cond ], [ %add_sum, %polly.loop_exit ]
  %merge.lb = phi i32 [ 0, %polly.cond ], [ %leftover_lb, %polly.loop_exit ]
  %polly.loop_guard17 = icmp slt i32 %merge.lb, %count
  br i1 %polly.loop_guard17, label %polly.loop_preheader15, label %for.end

polly.stmt.for.body22:
  %p_add30 = phi i32 [ %p_add34, %polly.loop_preheader15 ], [ %p_add, %polly.stmt.for.body22 ]
  %polly.indvar18 = phi i32 [ %merge.lb, %polly.loop_preheader15 ], [ %polly.indvar_next19, %polly.stmt.for.body22 ]
  %5 = tail call i32 @llvm.annotation.i32(i32 %polly.indvar18, ptr null, ptr null, i32 0), !polly.loop.smallTripCount !0
  %scevgep23 = getelementptr [1000 x i32], ptr @A, i32 0, i32 %polly.indvar18
  %_p_scalar_ = load i32, ptr %scevgep23, align 4
  %scevgep24 = getelementptr [1000 x i32], ptr @B, i32 0, i32 %polly.indvar18
  %_p_scalar_25 = load i32, ptr %scevgep24, align 4
  %p_mul = mul nsw i32 %_p_scalar_25, %_p_scalar_
  %p_add = add nsw i32 %p_mul, %p_add30
  %polly.indvar_next19 = add nsw i32 %polly.indvar18, 1
  %polly.loop_cond21 = icmp slt i32 %polly.indvar18, %polly.adjust_ub20
  br i1 %polly.loop_cond21, label %polly.stmt.for.body22, label %for.end.loopexit

polly.loop_preheader15:
  %polly.adjust_ub20 = add i32 %count, -1
  br label %polly.stmt.for.body22
}

declare i32 @llvm.annotation.i32(i32, ptr, ptr, i32) #1

!0 = !{}
