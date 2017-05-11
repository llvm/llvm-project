; RUN: opt -S -polly-detect-fortran-arrays -analyze -polly-process-unprofitable \
; RUN: -polly-remarks-minimal -polly-canonicalize -polly-scops \
; RUN: -polly-dependences -polly-canonicalize \
; RUN: -polly-allow-nonaffine -polly-ignore-aliasing \
; RUN: -polly-invariant-load-hoisting < %s| FileCheck %s
;
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
  %0 = load i32, i32* %n, align 4, !tbaa !0
  %1 = icmp sgt i32 %0, 21
  br i1 %1, label %"3.preheader", label %return

"3.preheader":                                    ; preds = %entry
  %2 = load i64, i64* getelementptr inbounds (%"struct.array1_real(kind=8)", %"struct.array1_real(kind=8)"* @__src_soil_MOD_xdzs, i64 0, i32 1), align 8, !tbaa !3
  %3 = load i8*, i8** getelementptr inbounds (%"struct.array1_real(kind=8)", %"struct.array1_real(kind=8)"* @__src_soil_MOD_xdzs, i64 0, i32 0), align 32, !tbaa !5
  %4 = bitcast i8* %3 to double*
  %5 = add i32 %0, 1
  br label %"3"

"3":                                              ; preds = %"3", %"3.preheader"
  %indvars.iv = phi i64 [ 22, %"3.preheader" ], [ %indvars.iv.next, %"3" ]
  %6 = add nsw i64 %indvars.iv, %2
  %7 = getelementptr inbounds double, double* %4, i64 %6
  %8 = load double, double* %7, align 8, !tbaa !7
  %9 = fmul double %8, %8
  %10 = add nsw i64 %indvars.iv, -1
  %11 = add nsw i64 %10, %2
  %12 = getelementptr inbounds double, double* %4, i64 %11
  %13 = load double, double* %12, align 8, !tbaa !7
  %14 = fadd double %9, %13
  store double %14, double* %7, align 8, !tbaa !7
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %5
  br i1 %exitcond, label %return, label %"3"

return:                                           ; preds = %"3", %entry
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
