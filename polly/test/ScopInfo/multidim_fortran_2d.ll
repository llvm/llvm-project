; RUN: opt %loadPolly -polly-scops -analyze \
; RUN: -polly-invariant-load-hoisting=true < %s | FileCheck %s

;   subroutine init_array(ni, nj, pi, pj, a)
;   implicit none
;
;   double precision, dimension(nj, ni) :: a
;   integer*8 :: ni, nj
;   integer*8 :: pi, pj
;   integer*8 :: i, j
;
;   do i = 1, ni
;     do j = 1, nj
;       a(j, i) = i + j
;     end do
;   end do
;   end subroutine
;
;  Verify we correctly delinearize accesses

; CHECK: MustWriteAccess :=  [Reduction Type: NONE] [Scalar: 0]
; CHECK:   [tmp9, tmp14] -> { Stmt_bb17[i0, i1] -> MemRef_arg4[i0, i1] };

target datalayout = "e-p:64:64:64-S128-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f16:16:16-f32:32:32-f64:64:64-f128:128:128-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"

define void @ham(i64* noalias %arg, i64* noalias %arg1, i64* noalias %arg2, i64* noalias %arg3, [0 x double]* noalias %arg4) unnamed_addr {
bb:
  br label %bb5

bb5:                                              ; preds = %bb
  %tmp = load i64, i64* %arg1, align 8
  %tmp6 = icmp slt i64 %tmp, 0
  %tmp7 = select i1 %tmp6, i64 0, i64 %tmp
  %tmp8 = xor i64 %tmp7, -1
  %tmp9 = load i64, i64* %arg, align 8
  %tmp10 = icmp sgt i64 %tmp9, 0
  br i1 %tmp10, label %bb11, label %bb32

bb11:                                             ; preds = %bb5
  br label %bb12

bb12:                                             ; preds = %bb28, %bb11
  %tmp13 = phi i64 [ %tmp30, %bb28 ], [ 1, %bb11 ]
  %tmp14 = load i64, i64* %arg1, align 8
  %tmp15 = icmp sgt i64 %tmp14, 0
  br i1 %tmp15, label %bb16, label %bb28

bb16:                                             ; preds = %bb12
  br label %bb17

bb17:                                             ; preds = %bb17, %bb16
  %tmp18 = phi i64 [ %tmp26, %bb17 ], [ 1, %bb16 ]
  %tmp19 = mul i64 %tmp13, %tmp7
  %tmp20 = add i64 %tmp19, %tmp8
  %tmp21 = add i64 %tmp20, %tmp18
  %tmp22 = add i64 %tmp18, %tmp13
  %tmp23 = sitofp i64 %tmp22 to double
  %tmp24 = getelementptr [0 x double], [0 x double]* %arg4, i64 0, i64 %tmp21
  store double %tmp23, double* %tmp24, align 8
  %tmp25 = icmp eq i64 %tmp18, %tmp14
  %tmp26 = add i64 %tmp18, 1
  br i1 %tmp25, label %bb27, label %bb17

bb27:                                             ; preds = %bb17
  br label %bb28

bb28:                                             ; preds = %bb27, %bb12
  %tmp29 = icmp eq i64 %tmp13, %tmp9
  %tmp30 = add i64 %tmp13, 1
  br i1 %tmp29, label %bb31, label %bb12

bb31:                                             ; preds = %bb28
  br label %bb32

bb32:                                             ; preds = %bb31, %bb5
  ret void
}
