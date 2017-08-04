; Check that the runtime size computation is generated for Fortran arrays.

; Regular code generation backend:
; RUN: opt %loadPolly -S -polly-detect-fortran-arrays \
; RUN: -polly-codegen < %s | FileCheck %s

; What the input fortran code should look like. NOTE: this is fake, the
; .ll file was hand-written.
;
; MODULE testmod
; USE data_parameters, ONLY : &
; IMPLICIT NONE
;
; INTEGER (KIND=iintegers), ALLOCATABLE, PRIVATE  :: &
;   arrin(:), arrout(:)
; CONTAINS
;
; SUBROUTINE test()
;   INTEGER (KIND=iintegers) :: i
;
;   DO i = 1, 100
;       arrout(i) = arrin(i) * arrin(i)
;   END DO
; END SUBROUTINE test
; END MODULE testmod

target datalayout = "e-p:64:64:64-S128-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i32:64:64-f16:16:16-f32:32:32-f64:64:64-f128:128:128-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-unknown-linux-gnu"

module asm "\09.ident\09\22GCC: (GNU) 4.6.4 LLVM: 3.3.1\22"

%"struct.array1_real(kind=8)" = type { i8*, i32, i32, [1 x %struct.descriptor_dimension] }
%struct.descriptor_dimension = type { i32, i32, i32 }

@arrin = unnamed_addr global %"struct.array1_real(kind=8)" zeroinitializer, align 32
@arrout = unnamed_addr global %"struct.array1_real(kind=8)" zeroinitializer, align 32

; Function Attrs: nounwind uwtable
define void @__src_soil_MOD_terra1() unnamed_addr #0 {
entry:
  br label %entry.split

entry.split:                                      ; preds = %entry
  %rawmemin1 = load i32*, i32** bitcast (%"struct.array1_real(kind=8)"* @arrin to i32**), align 32, !tbaa !0
  %rawmemout2 = load i32*, i32** bitcast (%"struct.array1_real(kind=8)"* @arrout to i32**), align 32, !tbaa !0
  br label %for.body

for.body:                                         ; preds = %entry.split, %for.body
  %indvars.iv = phi i64 [ 1, %entry.split ], [ %indvars.iv.next4, %for.body ]
  %inslot = getelementptr inbounds i32, i32* %rawmemin1, i64 %indvars.iv
  %inval = load i32, i32* %inslot, align 8
  %outslot = getelementptr inbounds i32, i32* %rawmemout2, i64 %indvars.iv
  %out = mul nsw i32 %inval, %inval
  store i32 %out, i32* %outslot, align 8
  %indvars.iv.next4 = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next4, 100
  br i1 %exitcond, label %return, label %for.body

return:                                           ; preds = %for.body
  ret void
}

attributes #0 = { nounwind uwtable }

!0 = !{!1, !1, i32 0}
!1 = !{!"alias set 3: void*", !2}
!2 = distinct !{!2}


; CHECK:       %MemRef_rawmemin1_end = load i32, i32* getelementptr inbounds (%"struct.array1_real(kind=8)", %"struct.array1_real(kind=8)"* @arrin, i64 0, i32 3, i64 0, i32 2)
; CHECK-NEXT:  %MemRef_rawmemin1_begin = load i32, i32* getelementptr inbounds (%"struct.array1_real(kind=8)", %"struct.array1_real(kind=8)"* @arrin, i64 0, i32 3, i64 0, i32 1)
; CHECK-NEXT:  %MemRef_rawmemin1_end_begin_delta = sub nsw i32 %MemRef_rawmemin1_end, %MemRef_rawmemin1_begin
; CHECK-NEXT:  %MemRef_rawmemin1_size = add nsw i32 %MemRef_rawmemin1_end, 1
; CHECK-NEXT:  %MemRef_rawmemout2_end = load i32, i32* getelementptr inbounds (%"struct.array1_real(kind=8)", %"struct.array1_real(kind=8)"* @arrout, i64 0, i32 3, i64 0, i32 2)
; CHECK-NEXT:  %MemRef_rawmemout2_begin = load i32, i32* getelementptr inbounds (%"struct.array1_real(kind=8)", %"struct.array1_real(kind=8)"* @arrout, i64 0, i32 3, i64 0, i32 1)
; CHECK-NEXT:  %MemRef_rawmemout2_end_begin_delta = sub nsw i32 %MemRef_rawmemout2_end, %MemRef_rawmemout2_begin
; CHECK-NEXT:  %MemRef_rawmemout2_size = add nsw i32 %MemRef_rawmemout2_end, 1
