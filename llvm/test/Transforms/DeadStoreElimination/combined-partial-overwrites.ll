; RUN: opt -S -dse < %s | FileCheck %s
target datalayout = "E-m:e-i64:64-n32:64"
target triple = "powerpc64-bgq-linux"

%"struct.std::complex" = type { { float, float } }

define void @_Z4testSt7complexIfE(%"struct.std::complex"* noalias nocapture sret %agg.result, i64 %c.coerce) {
entry:
; CHECK-LABEL: @_Z4testSt7complexIfE

  %ref.tmp = alloca i64, align 8
  %tmpcast = bitcast i64* %ref.tmp to %"struct.std::complex"*
  %c.sroa.0.0.extract.shift = lshr i64 %c.coerce, 32
  %c.sroa.0.0.extract.trunc = trunc i64 %c.sroa.0.0.extract.shift to i32
  %0 = bitcast i32 %c.sroa.0.0.extract.trunc to float
  %c.sroa.2.0.extract.trunc = trunc i64 %c.coerce to i32
  %1 = bitcast i32 %c.sroa.2.0.extract.trunc to float
  call void @_Z3barSt7complexIfE(%"struct.std::complex"* nonnull sret %tmpcast, i64 %c.coerce)
  %2 = bitcast %"struct.std::complex"* %agg.result to i64*
  %3 = load i64, i64* %ref.tmp, align 8
  store i64 %3, i64* %2, align 4
; CHECK-NOT: store i64

  %_M_value.realp.i.i = getelementptr inbounds %"struct.std::complex", %"struct.std::complex"* %agg.result, i64 0, i32 0, i32 0
  %4 = lshr i64 %3, 32
  %5 = trunc i64 %4 to i32
  %6 = bitcast i32 %5 to float
  %_M_value.imagp.i.i = getelementptr inbounds %"struct.std::complex", %"struct.std::complex"* %agg.result, i64 0, i32 0, i32 1
  %7 = trunc i64 %3 to i32
  %8 = bitcast i32 %7 to float
  %mul_ad.i.i = fmul fast float %6, %1
  %mul_bc.i.i = fmul fast float %8, %0
  %mul_i.i.i = fadd fast float %mul_ad.i.i, %mul_bc.i.i
  %mul_ac.i.i = fmul fast float %6, %0
  %mul_bd.i.i = fmul fast float %8, %1
  %mul_r.i.i = fsub fast float %mul_ac.i.i, %mul_bd.i.i
  store float %mul_r.i.i, float* %_M_value.realp.i.i, align 4
  store float %mul_i.i.i, float* %_M_value.imagp.i.i, align 4
  ret void
; CHECK: ret void
}

declare void @_Z3barSt7complexIfE(%"struct.std::complex"* sret, i64)

define void @test1(i32 *%ptr) {
entry:
; CHECK-LABEL: @test1

 store i32 5, i32* %ptr
 %bptr = bitcast i32* %ptr to i8*
 store i8 7, i8* %bptr
 %wptr = bitcast i32* %ptr to i16*
 store i16 -30062, i16* %wptr
 %bptr2 = getelementptr inbounds i8, i8* %bptr, i64 2
 store i8 25, i8* %bptr2
 %bptr3 = getelementptr inbounds i8, i8* %bptr, i64 3
 store i8 47, i8* %bptr3
 %bptr1 = getelementptr inbounds i8, i8* %bptr, i64 1
 %wptrp = bitcast i8* %bptr1 to i16*
 store i16 2020, i16* %wptrp, align 1
 ret void

; CHECK-NOT: store i32 5, i32* %ptr
; CHECK-NOT: store i8 7, i8* %bptr
; CHECK: store i16 -30062, i16* %wptr
; CHECK-NOT: store i8 25, i8* %bptr2
; CHECK: store i8 47, i8* %bptr3
; CHECK: store i16 2020, i16* %wptrp, align 1

; CHECK: ret void
}

define void @test2(i32 *%ptr) {
entry:
; CHECK-LABEL: @test2

  store i32 5, i32* %ptr

  %bptr = bitcast i32* %ptr to i8*
  %bptrm1 = getelementptr inbounds i8, i8* %bptr, i64 -1
  %bptr1 = getelementptr inbounds i8, i8* %bptr, i64 1
  %bptr2 = getelementptr inbounds i8, i8* %bptr, i64 2
  %bptr3 = getelementptr inbounds i8, i8* %bptr, i64 3

  %wptr = bitcast i8* %bptr to i16*
  %wptrm1 = bitcast i8* %bptrm1 to i16*
  %wptr1 = bitcast i8* %bptr1 to i16*
  %wptr2 = bitcast i8* %bptr2 to i16*
  %wptr3 = bitcast i8* %bptr3 to i16*

  store i16 1456, i16* %wptrm1, align 1
  store i16 1346, i16* %wptr, align 1
  store i16 1756, i16* %wptr1, align 1
  store i16 1126, i16* %wptr2, align 1
  store i16 5656, i16* %wptr3, align 1

; CHECK-NOT: store i32 5, i32* %ptr

; CHECK: store i16 1456, i16* %wptrm1, align 1
; CHECK: store i16 1346, i16* %wptr, align 1
; CHECK: store i16 1756, i16* %wptr1, align 1
; CHECK: store i16 1126, i16* %wptr2, align 1
; CHECK: store i16 5656, i16* %wptr3, align 1

  ret void

; CHECK: ret void
}

define signext i8 @test3(i32 *%ptr) {
entry:
; CHECK-LABEL: @test3

  store i32 5, i32* %ptr

  %bptr = bitcast i32* %ptr to i8*
  %bptrm1 = getelementptr inbounds i8, i8* %bptr, i64 -1
  %bptr1 = getelementptr inbounds i8, i8* %bptr, i64 1
  %bptr2 = getelementptr inbounds i8, i8* %bptr, i64 2
  %bptr3 = getelementptr inbounds i8, i8* %bptr, i64 3

  %wptr = bitcast i8* %bptr to i16*
  %wptrm1 = bitcast i8* %bptrm1 to i16*
  %wptr1 = bitcast i8* %bptr1 to i16*
  %wptr2 = bitcast i8* %bptr2 to i16*
  %wptr3 = bitcast i8* %bptr3 to i16*

  %v = load i8, i8* %bptr, align 1
  store i16 1456, i16* %wptrm1, align 1
  store i16 1346, i16* %wptr, align 1
  store i16 1756, i16* %wptr1, align 1
  store i16 1126, i16* %wptr2, align 1
  store i16 5656, i16* %wptr3, align 1

; CHECK: store i32 5, i32* %ptr

  ret i8 %v

; CHECK: ret i8 %v
}

%struct.foostruct = type {
i32 (i8*, i8**, i32, i8, i8*)*,
i32 (i8*, i8**, i32, i8, i8*)*,
i32 (i8*, i8**, i32, i8, i8*)*,
i32 (i8*, i8**, i32, i8, i8*)*,
void (i8*, i32, i32)*
}
declare void @llvm.memset.p0i8.i64(i8* nocapture, i8, i64, i32, i1)
declare void @goFunc(%struct.foostruct*)
declare i32 @fa(i8*, i8**, i32, i8, i8*)

define void @test4()  {
entry:
; CHECK-LABEL: @test4

  %bang = alloca %struct.foostruct, align 8
  %v1 = bitcast %struct.foostruct* %bang to i8*
  call void @llvm.memset.p0i8.i64(i8* %v1, i8 0, i64 40, i32 8, i1 false)
  %v2 = getelementptr inbounds %struct.foostruct, %struct.foostruct* %bang, i64 0, i32 0
  store i32 (i8*, i8**, i32, i8, i8*)* @fa, i32 (i8*, i8**, i32, i8, i8*)** %v2, align 8
  %v3 = getelementptr inbounds %struct.foostruct, %struct.foostruct* %bang, i64 0, i32 1
  store i32 (i8*, i8**, i32, i8, i8*)* @fa, i32 (i8*, i8**, i32, i8, i8*)** %v3, align 8
  %v4 = getelementptr inbounds %struct.foostruct, %struct.foostruct* %bang, i64 0, i32 2
  store i32 (i8*, i8**, i32, i8, i8*)* @fa, i32 (i8*, i8**, i32, i8, i8*)** %v4, align 8
  %v5 = getelementptr inbounds %struct.foostruct, %struct.foostruct* %bang, i64 0, i32 3
  store i32 (i8*, i8**, i32, i8, i8*)* @fa, i32 (i8*, i8**, i32, i8, i8*)** %v5, align 8
  %v6 = getelementptr inbounds %struct.foostruct, %struct.foostruct* %bang, i64 0, i32 4
  store void (i8*, i32, i32)* null, void (i8*, i32, i32)** %v6, align 8
  call void @goFunc(%struct.foostruct* %bang)
  ret void

; CHECK-NOT: memset
; CHECK: ret void
}

define signext i8 @test5(i32 *%ptr) {
entry:
; CHECK-LABEL: @test5

  store i32 0, i32* %ptr

  %bptr = bitcast i32* %ptr to i8*
  %bptr1 = getelementptr inbounds i8, i8* %bptr, i64 1
  %bptr2 = getelementptr inbounds i8, i8* %bptr, i64 2
  %bptr3 = getelementptr inbounds i8, i8* %bptr, i64 3

  %wptr = bitcast i8* %bptr to i16*
  %wptr1 = bitcast i8* %bptr1 to i16*
  %wptr2 = bitcast i8* %bptr2 to i16*

  store i16 65535, i16* %wptr2, align 1
  store i16 1456, i16* %wptr1, align 1
  store i16 1346, i16* %wptr, align 1

; CHECK-NOT: store i32 0, i32* %ptr

  ret i8 0
}

