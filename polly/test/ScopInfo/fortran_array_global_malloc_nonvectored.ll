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
; SUBROUTINE terra1(n)
;   INTEGER, intent(in) :: n
;   INTEGER (KIND=iintegers) ::  &
;     j
;   Allocate(xdzs(n));
;    DO j = 2, n
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
@.cst = private unnamed_addr constant [67 x i8] c"Integer overflow when calculating the amount of memory to allocate\00", align 64
@.cst1 = private unnamed_addr constant [37 x i8] c"Allocation would exceed memory limit\00", align 64
@.cst2 = private unnamed_addr constant [93 x i8] c"At line 23 of file /home/siddhart/cosmo-self-installation/cosmo-pompa/cosmo/src/src_soil.f90\00", align 64
@.cst3 = private unnamed_addr constant [55 x i8] c"Attempting to allocate already allocated variable '%s'\00", align 64
@.cst4 = private unnamed_addr constant [5 x i8] c"xdzs\00", align 8

; Function Attrs: nounwind uwtable
define void @__src_soil_MOD_terra1(i32* noalias nocapture %n) unnamed_addr #0 {
entry:
  br label %entry.split

entry.split:                                      ; preds = %entry
  store i64 537, i64* getelementptr inbounds (%"struct.array1_real(kind=8)", %"struct.array1_real(kind=8)"* @__src_soil_MOD_xdzs, i64 0, i32 2), align 16, !tbaa !0
  store i64 1, i64* getelementptr inbounds (%"struct.array1_real(kind=8)", %"struct.array1_real(kind=8)"* @__src_soil_MOD_xdzs, i64 0, i32 3, i64 0, i32 1), align 8, !tbaa !0
  %tmp = load i32, i32* %n, align 4, !tbaa !3
  %tmp1 = sext i32 %tmp to i64
  store i64 %tmp1, i64* getelementptr inbounds (%"struct.array1_real(kind=8)", %"struct.array1_real(kind=8)"* @__src_soil_MOD_xdzs, i64 0, i32 3, i64 0, i32 2), align 8, !tbaa !0
  store i64 1, i64* getelementptr inbounds (%"struct.array1_real(kind=8)", %"struct.array1_real(kind=8)"* @__src_soil_MOD_xdzs, i64 0, i32 3, i64 0, i32 0), align 8, !tbaa !0
  %tmp2 = icmp slt i32 %tmp, 1
  %tmp3 = zext i32 %tmp to i64
  %tmp4 = shl nuw nsw i64 %tmp3, 3
  %.24 = select i1 %tmp2, i64 0, i64 %tmp4
  %tmp5 = icmp ne i64 %.24, 0
  %tmp6 = select i1 %tmp5, i64 %.24, i64 1
  %tmp7 = tail call noalias i8* @malloc(i64 %tmp6) #2
  store i8* %tmp7, i8** getelementptr inbounds (%"struct.array1_real(kind=8)", %"struct.array1_real(kind=8)"* @__src_soil_MOD_xdzs, i64 0, i32 0), align 32, !tbaa !5
  store i64 -1, i64* getelementptr inbounds (%"struct.array1_real(kind=8)", %"struct.array1_real(kind=8)"* @__src_soil_MOD_xdzs, i64 0, i32 1), align 8, !tbaa !0
  %tmp8 = icmp sgt i32 %tmp, 1
  br i1 %tmp8, label %"21.preheader", label %return

"21.preheader":                                   ; preds = %entry.split
  %tmp9 = bitcast i8* %tmp7 to double*
  %tmp10 = add i32 %tmp, 1
  br label %"21"

"21":                                             ; preds = %"21", %"21.preheader"
  %tmp11 = phi double [ undef, %"21.preheader" ], [ %tmp16, %"21" ]
  %indvars.iv = phi i64 [ 2, %"21.preheader" ], [ %indvars.iv.next, %"21" ]
  %tmp12 = add nsw i64 %indvars.iv, -1
  %tmp13 = getelementptr inbounds double, double* %tmp9, i64 %tmp12
  %tmp14 = load double, double* %tmp13, align 8, !tbaa !7
  %tmp15 = fmul double %tmp14, %tmp14
  %tmp16 = fadd double %tmp11, %tmp15
  store double %tmp16, double* %tmp13, align 8, !tbaa !7
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %lftr.wideiv1 = trunc i64 %indvars.iv.next to i32
  %exitcond2 = icmp eq i32 %lftr.wideiv1, %tmp10
  br i1 %exitcond2, label %return.loopexit, label %"21"

return.loopexit:                                  ; preds = %"21"
  br label %return

return:                                           ; preds = %return.loopexit, %entry.split
  ret void
}

; Function Attrs: noreturn
declare void @_gfortran_runtime_error(i8*, ...) #1

; Function Attrs: nounwind
declare noalias i8* @malloc(i64) #2

; Function Attrs: noreturn
declare void @_gfortran_os_error(i8*) #1

; Function Attrs: noreturn
declare void @_gfortran_runtime_error_at(i8*, i8*, ...) #1

attributes #0 = { nounwind uwtable }
attributes #1 = { noreturn }
attributes #2 = { nounwind }

!0 = !{!1, !1, i64 0}
!1 = !{!"alias set 4: integer(kind=8)", !2}
!2 = distinct !{!2}
!3 = !{!4, !4, i64 0}
!4 = !{!"alias set 11: integer(kind=4)", !2}
!5 = !{!6, !6, i64 0}
!6 = !{!"alias set 3: void*", !2}
!7 = !{!8, !8, i64 0}
!8 = !{!"alias set 18: real(kind=8)", !2}

; CHECK: ReadAccess :=	[Reduction Type: NONE] [Fortran array descriptor: __src_soil_MOD_xdzs] [Scalar: 0]
; CHECK: MustWriteAccess :=	[Reduction Type: NONE] [Fortran array descriptor: __src_soil_MOD_xdzs] [Scalar: 0]
