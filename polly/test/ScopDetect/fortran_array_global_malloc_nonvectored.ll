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
  store i64 537, i64* getelementptr inbounds (%"struct.array1_real(kind=8)", %"struct.array1_real(kind=8)"* @__src_soil_MOD_xdzs, i64 0, i32 2), align 16, !tbaa !0
  store i64 1, i64* getelementptr inbounds (%"struct.array1_real(kind=8)", %"struct.array1_real(kind=8)"* @__src_soil_MOD_xdzs, i64 0, i32 3, i64 0, i32 1), align 8, !tbaa !0
  %0 = load i32, i32* %n, align 4, !tbaa !3
  %1 = sext i32 %0 to i64
  store i64 %1, i64* getelementptr inbounds (%"struct.array1_real(kind=8)", %"struct.array1_real(kind=8)"* @__src_soil_MOD_xdzs, i64 0, i32 3, i64 0, i32 2), align 8, !tbaa !0
  store i64 1, i64* getelementptr inbounds (%"struct.array1_real(kind=8)", %"struct.array1_real(kind=8)"* @__src_soil_MOD_xdzs, i64 0, i32 3, i64 0, i32 0), align 8, !tbaa !0
  %2 = icmp slt i32 %0, 0
  %3 = select i1 %2, i64 0, i64 %1
  %4 = icmp eq i64 %3, 0
  br i1 %4, label %"16", label %"8"

"8":                                              ; preds = %entry
  %5 = sdiv i64 9223372036854775807, %1
  %6 = icmp slt i64 %5, 1
  %7 = icmp slt i32 %0, 1
  %8 = shl nsw i64 %3, 3
  %.2 = select i1 %7, i64 0, i64 %8
  br i1 %6, label %"15", label %"16"

"15":                                             ; preds = %"8"

  unreachable

"16":                                             ; preds = %"8", %entry
  %.24 = phi i64 [ %.2, %"8" ], [ 0, %entry ]
  %9 = load i8*, i8** getelementptr inbounds (%"struct.array1_real(kind=8)", %"struct.array1_real(kind=8)"* @__src_soil_MOD_xdzs, i64 0, i32 0), align 32, !tbaa !5
  %10 = icmp eq i8* %9, null
  br i1 %10, label %"17", label %"20"

"17":                                             ; preds = %"16"
  %11 = icmp ne i64 %.24, 0
  %12 = select i1 %11, i64 %.24, i64 1
  %13 = tail call noalias i8* @malloc(i64 %12) #2 ;<= 1. malloc
  %14 = icmp eq i8* %13, null
  br i1 %14, label %"18", label %"19"

"18":                                             ; preds = %"17"
  unreachable

"19":                                             ; preds = %"17"
  store i8* %13, i8** getelementptr inbounds (%"struct.array1_real(kind=8)", %"struct.array1_real(kind=8)"* @__src_soil_MOD_xdzs, i64 0, i32 0), align 32, !tbaa !5
  store i64 -1, i64* getelementptr inbounds (%"struct.array1_real(kind=8)", %"struct.array1_real(kind=8)"* @__src_soil_MOD_xdzs, i64 0, i32 1), align 8, !tbaa !0
  %15 = icmp sgt i32 %0, 1
  br i1 %15, label %"21.preheader", label %return

"21.preheader":                                   ; preds = %"19"
  %16 = bitcast i8* %13 to double* ;<= 2. bitcast to double*
  %17 = add i32 %0, 1
  br label %"21"

"20":                                             ; preds = %"16"
  unreachable

"21":                                             ; preds = %"21", %"21.preheader"
  %18 = phi double [ undef, %"21.preheader" ], [ %23, %"21" ]
  %indvars.iv = phi i64 [ 2, %"21.preheader" ], [ %indvars.iv.next, %"21" ]
  %19 = add nsw i64 %indvars.iv, -1
  %20 = getelementptr inbounds double, double* %16, i64 %19 ;<= 3. GEP
  %21 = load double, double* %20, align 8, !tbaa !7
  %22 = fmul double %21, %21
  %23 = fadd double %22, %18
  store double %23, double* %20, align 8, !tbaa !7 ;<= 4. store
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %17
  br i1 %exitcond, label %return, label %"21"

return:                                           ; preds = %"21", %"19"
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
attributes #3 = { noreturn nounwind }

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
