; RUN: opt %loadNPMPolly '-passes=print<polly-function-scops>' -disable-output \
; RUN: -polly-precise-fold-accesses \
; RUN: -polly-invariant-load-hoisting=true < %s 2>&1 | FileCheck %s

;   subroutine init_array(ni, nj, pi, pj, a)
;   implicit none

;   double precision, dimension(nj, ni) :: a
;   integer*8 :: ni, nj
;   integer*8 :: pi, pj
;   integer*8 :: i, j

;   do i = 1, ni
;     do j = 1, nj
;       a(j + pi, i + pj) = i + j
;     end do
;   end do
;   end subroutine

; CHECK: [tmp9, nj_loaded2, tmp20, tmp19] -> { Stmt_bb17[i0, i1] -> MemRef_a[-1 + tmp20 + i0, nj_loaded2 + tmp19 + i1] : i1 < -tmp19; Stmt_bb17[i0, i1] -> MemRef_a[tmp20 + i0, tmp19 + i1] : i1 >= -tmp19 };


target datalayout = "e-p:64:64:64-S128-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f16:16:16-f32:32:32-f64:64:64-f128:128:128-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-unknown-linux-gnu"

module asm "\09.ident\09\22GCC: (GNU) 4.6.4 LLVM: 3.3.1\22"

; Function Attrs: nounwind uwtable
define void @blam(ptr noalias %arg, ptr noalias %nj, ptr noalias %arg2, ptr noalias %arg3, ptr noalias %a) unnamed_addr #0 {
bb:
  br label %bb5

bb5:                                              ; preds = %bb
  %nj_loaded = load i64, ptr %nj, align 8
  %tmp6 = icmp slt i64 %nj_loaded, 0
  %tmp7 = select i1 %tmp6, i64 0, i64 %nj_loaded
  %tmp8 = xor i64 %tmp7, -1
  %tmp9 = load i64, ptr %arg, align 8
  %tmp10 = icmp sgt i64 %tmp9, 0
  br i1 %tmp10, label %bb11, label %bb36

bb11:                                             ; preds = %bb5
  br label %bb12

bb12:                                             ; preds = %bb32, %bb11
  %tmp13 = phi i64 [ %tmp34, %bb32 ], [ 1, %bb11 ]
  %nj_loaded2 = load i64, ptr %nj, align 8
  %tmp15 = icmp sgt i64 %nj_loaded2, 0
  br i1 %tmp15, label %bb16, label %bb32

bb16:                                             ; preds = %bb12
  br label %bb17

bb17:                                             ; preds = %bb17, %bb16
  %tmp18 = phi i64 [ %tmp30, %bb17 ], [ 1, %bb16 ]
  %tmp19 = load i64, ptr %arg2, align 8
  %tmp20 = load i64, ptr %arg3, align 8
  %tmp21 = add i64 %tmp20, %tmp13
  %tmp22 = mul i64 %tmp21, %tmp7
  %tmp23 = add i64 %tmp18, %tmp8
  %tmp24 = add i64 %tmp23, %tmp19
  %tmp25 = add i64 %tmp24, %tmp22
  %tmp26 = add i64 %tmp18, %tmp13
  %tmp27 = sitofp i64 %tmp26 to double
  %tmp28 = getelementptr [0 x double], ptr %a, i64 0, i64 %tmp25
  store double %tmp27, ptr %tmp28, align 8
  %tmp29 = icmp eq i64 %tmp18, %nj_loaded2
  %tmp30 = add i64 %tmp18, 1
  br i1 %tmp29, label %bb31, label %bb17

bb31:                                             ; preds = %bb17
  br label %bb32

bb32:                                             ; preds = %bb31, %bb12
  %tmp33 = icmp eq i64 %tmp13, %tmp9
  %tmp34 = add i64 %tmp13, 1
  br i1 %tmp33, label %bb35, label %bb12

bb35:                                             ; preds = %bb32
  br label %bb36

bb36:                                             ; preds = %bb35, %bb5
  ret void
}

attributes #0 = { nounwind uwtable }
