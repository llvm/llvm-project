; RUN: opt %loadPolly -analyze -S -polly-detect-fortran-arrays \
; RUN: -polly-process-unprofitable -polly-scops  < %s | FileCheck %s

; MODULE src_soil
; USE data_parameters, ONLY :   &
;     wp,        & ! KIND-type parameter for real variables
;     iintegers    ! KIND-type parameter for standard integer variables
; IMPLICIT NONE
; REAL (KIND = wp),     ALLOCATABLE, PRIVATE  :: &
;   xdzs     (:)
; CONTAINS
;
; SUBROUTINE terra1(n)
;   INTEGER, intent(in) :: n
;
;   INTEGER (KIND=iintegers) ::  &
;     j
;
;    DO j = 22, n
;         xdzs(j) = xdzs(j) * xdzs(j) + xdzs(j - 1)
;   END DO
; END SUBROUTINE terra1
; END MODULE src_soil

target datalayout = "e-p:64:64:64-S128-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f16:16:16-f32:32:32-f64:64:64-f128:128:128-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-unknown-linux-gnu"

module asm "\09.ident\09\22GCC: (GNU) 4.6.4 LLVM: 3.3.1\22"

%"struct.array1_real(kind=8)" = type { i8*, i64, i64, [1 x %struct.descriptor_dimension] }
%struct.descriptor_dimension = type { i64, i64, i64 }

@__src_soil_MOD_xdzs = unnamed_addr global %"struct.array1_real(kind=8)" zeroinitializer, align 32

; Function Attrs: nounwind uwtable
define void @__src_soil_MOD_terra1(i32* noalias nocapture %n) unnamed_addr #0 {
entry:
  br label %entry.split

entry.split:                                      ; preds = %entry
  %tmp = load i32, i32* %n, align 4, !tbaa !0
  %tmp1 = icmp sgt i32 %tmp, 21
  br i1 %tmp1, label %"3.preheader", label %return

"3.preheader":                                    ; preds = %entry.split
  %tmp2 = load i64, i64* getelementptr inbounds (%"struct.array1_real(kind=8)", %"struct.array1_real(kind=8)"* @__src_soil_MOD_xdzs, i64 0, i32 1), align 8, !tbaa !3
  %tmp3 = load double*, double** bitcast (%"struct.array1_real(kind=8)"* @__src_soil_MOD_xdzs to double**), align 32, !tbaa !5
  %tmp4 = add i32 %tmp, 1
  br label %"3"

"3":                                              ; preds = %"3", %"3.preheader"
  %indvars.iv = phi i64 [ 22, %"3.preheader" ], [ %indvars.iv.next, %"3" ]
  %tmp5 = add nsw i64 %indvars.iv, %tmp2
  %tmp6 = getelementptr inbounds double, double* %tmp3, i64 %tmp5
  %tmp7 = load double, double* %tmp6, align 8, !tbaa !7
  %tmp8 = fmul double %tmp7, %tmp7
  %tmp9 = add i64 %tmp2, -1
  %tmp10 = add i64 %tmp9, %indvars.iv
  %tmp11 = getelementptr inbounds double, double* %tmp3, i64 %tmp10
  %tmp12 = load double, double* %tmp11, align 8, !tbaa !7
  %tmp13 = fadd double %tmp8, %tmp12
  store double %tmp13, double* %tmp6, align 8, !tbaa !7
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %lftr.wideiv1 = trunc i64 %indvars.iv.next to i32
  %exitcond2 = icmp eq i32 %lftr.wideiv1, %tmp4
  br i1 %exitcond2, label %return.loopexit, label %"3"

return.loopexit:                                  ; preds = %"3"
  br label %return

return:                                           ; preds = %return.loopexit, %entry.split
  ret void
}

attributes #0 = { nounwind uwtable }

!0 = !{!1, !1, i64 0}
!1 = !{!"alias set 11: integer(kind=4)", !2}
!2 = distinct !{!2}
!3 = !{!4, !4, i64 0}
!4 = !{!"alias set 4: integer(kind=8)", !2}
!5 = !{!6, !6, i64 0}
!6 = !{!"alias set 3: void*", !2}
!7 = !{!8, !8, i64 0}
!8 = !{!"alias set 18: real(kind=8)", !2}

; CHECK: ReadAccess :=	[Reduction Type: NONE] [Fortran array descriptor: __src_soil_MOD_xdzs] [Scalar: 0]
; CHECK: ReadAccess :=	[Reduction Type: NONE] [Fortran array descriptor: __src_soil_MOD_xdzs] [Scalar: 0]
; CHECK: MustWriteAccess :=	[Reduction Type: NONE] [Fortran array descriptor: __src_soil_MOD_xdzs] [Scalar: 0]
