; RUN: opt < %s -scalarrepl -S | FileCheck %s
; Radar 7441282

target datalayout = "e-p:32:32:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-i64:32:32-f32:32:32-f64:32:32-v64:64:64-v128:128:128-a0:0:32-n32"
target triple = "thumbv7-apple-darwin10"

%struct.__neon_int16x8x2_t = type { <8 x i16>, <8 x i16> }
%struct.int16x8_t = type { <8 x i16> }
%struct.int16x8x2_t = type { [2 x %struct.int16x8_t] }
%union..0anon = type { %struct.int16x8x2_t }

define void @test(<8 x i16> %tmp.0, %struct.int16x8x2_t* %dst) nounwind {
; CHECK-LABEL: @test(
; CHECK-NOT: alloca
; CHECK: "alloca point"
; CHECK: store <8 x i16>
; CHECK: store <8 x i16>

entry:
  %tmp_addr = alloca %struct.int16x8_t
  %dst_addr = alloca %struct.int16x8x2_t*
  %__rv = alloca %union..0anon
  %__bx = alloca %struct.int16x8_t
  %__ax = alloca %struct.int16x8_t
  %tmp2 = alloca %struct.int16x8x2_t
  %0 = alloca %struct.int16x8x2_t
  %"alloca point" = bitcast i32 0 to i32
  %1 = getelementptr inbounds %struct.int16x8_t, %struct.int16x8_t* %tmp_addr, i32 0, i32 0
  store <8 x i16> %tmp.0, <8 x i16>* %1
  store %struct.int16x8x2_t* %dst, %struct.int16x8x2_t** %dst_addr
  %2 = getelementptr inbounds %struct.int16x8_t, %struct.int16x8_t* %__ax, i32 0, i32 0
  %3 = getelementptr inbounds %struct.int16x8_t, %struct.int16x8_t* %tmp_addr, i32 0, i32 0
  %4 = load <8 x i16>, <8 x i16>* %3, align 16
  store <8 x i16> %4, <8 x i16>* %2, align 16
  %5 = getelementptr inbounds %struct.int16x8_t, %struct.int16x8_t* %__bx, i32 0, i32 0
  %6 = getelementptr inbounds %struct.int16x8_t, %struct.int16x8_t* %tmp_addr, i32 0, i32 0
  %7 = load <8 x i16>, <8 x i16>* %6, align 16
  store <8 x i16> %7, <8 x i16>* %5, align 16
  %8 = getelementptr inbounds %struct.int16x8_t, %struct.int16x8_t* %__ax, i32 0, i32 0
  %9 = load <8 x i16>, <8 x i16>* %8, align 16
  %10 = getelementptr inbounds %struct.int16x8_t, %struct.int16x8_t* %__bx, i32 0, i32 0
  %11 = load <8 x i16>, <8 x i16>* %10, align 16
  %12 = getelementptr inbounds %union..0anon, %union..0anon* %__rv, i32 0, i32 0
  %13 = bitcast %struct.int16x8x2_t* %12 to %struct.__neon_int16x8x2_t*
  %14 = shufflevector <8 x i16> %9, <8 x i16> %11, <8 x i32> <i32 0, i32 8, i32 2, i32 10, i32 4, i32 12, i32 6, i32 14>
  %15 = getelementptr inbounds %struct.__neon_int16x8x2_t, %struct.__neon_int16x8x2_t* %13, i32 0, i32 0
  store <8 x i16> %14, <8 x i16>* %15
  %16 = shufflevector <8 x i16> %9, <8 x i16> %11, <8 x i32> <i32 1, i32 9, i32 3, i32 11, i32 5, i32 13, i32 7, i32 15>
  %17 = getelementptr inbounds %struct.__neon_int16x8x2_t, %struct.__neon_int16x8x2_t* %13, i32 0, i32 1
  store <8 x i16> %16, <8 x i16>* %17
  %18 = getelementptr inbounds %union..0anon, %union..0anon* %__rv, i32 0, i32 0
  %19 = bitcast %struct.int16x8x2_t* %0 to i8*
  %20 = bitcast %struct.int16x8x2_t* %18 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %19, i8* %20, i32 32, i1 false)
  %tmp21 = bitcast %struct.int16x8x2_t* %tmp2 to i8*
  %21 = bitcast %struct.int16x8x2_t* %0 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %tmp21, i8* %21, i32 32, i1 false)
  %22 = load %struct.int16x8x2_t*, %struct.int16x8x2_t** %dst_addr, align 4
  %23 = bitcast %struct.int16x8x2_t* %22 to i8*
  %tmp22 = bitcast %struct.int16x8x2_t* %tmp2 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %23, i8* %tmp22, i32 32, i1 false)
  br label %return

return:                                           ; preds = %entry
  ret void
}

; Radar 7466574
%struct._NSRange = type { i64 }

define void @test_memcpy_self() nounwind {
entry:
  %range = alloca %struct._NSRange
  br i1 undef, label %cond.true, label %cond.false

cond.true:                                        ; preds = %entry
  %tmp3 = bitcast %struct._NSRange* %range to i8*
  %tmp4 = bitcast %struct._NSRange* %range to i8*
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %tmp3, i8* %tmp4, i32 8, i1 false)
  ret void

cond.false:                                       ; preds = %entry
  ret void

; CHECK-LABEL: @test_memcpy_self(
; CHECK-NOT: alloca
; CHECK: br i1
}

declare void @llvm.memcpy.p0i8.p0i8.i32(i8* nocapture, i8* nocapture, i32, i1) nounwind
