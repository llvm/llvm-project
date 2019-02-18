; RUN: opt -memcpyopt -S < %s | FileCheck %s

declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture, i8* nocapture, i64, i1) nounwind

@undef = internal constant i32 undef, align 4
define void @test_undef() nounwind {
  %a = alloca i32, align 4
  %i8 = bitcast i32* %a to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %i8, i8* align 4 bitcast (i32* @undef to i8*), i64 4, i1 false)
  ret void
; CHECK-LABEL: @test_undef(
; CHECK:       call void @llvm.memset
; CHECK-NOT:   call void @llvm.memcpy
; CHECK:       ret void
}

@i32x3 = internal constant [3 x i32] [i32 -1, i32 -1, i32 -1], align 4
define void @test_i32x3() nounwind {
  %a = alloca [3 x i32], align 4
  %i8 = bitcast [3 x i32]* %a to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %i8, i8* align 4 bitcast ([3 x i32]* @i32x3 to i8*), i64 12, i1 false)
  ret void
; CHECK-LABEL: @test_i32x3(
; CHECK:       call void @llvm.memset
; CHECK-NOT:   call void @llvm.memcpy
; CHECK:       ret void
}

@i32x3_undef = internal constant [3 x i32] [i32 -1, i32 undef, i32 -1], align 4
define void @test_i32x3_undef() nounwind {
  %a = alloca [3 x i32], align 4
  %i8 = bitcast [3 x i32]* %a to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %i8, i8* align 4 bitcast ([3 x i32]* @i32x3_undef to i8*), i64 12, i1 false)
  ret void
; CHECK-LABEL: @test_i32x3_undef(
; CHECK:       call void @llvm.memset
; CHECK-NOT:   call void @llvm.memcpy
; CHECK:       ret void
}

%struct.bitfield = type { i8, [3 x i8] }
@bitfield = private unnamed_addr constant %struct.bitfield { i8 -86, [3 x i8] [i8 -86, i8 -86, i8 -86] }, align 4
define void @test_bitfield() nounwind {
  %a = alloca %struct.bitfield, align 4
  %i8 = bitcast %struct.bitfield* %a to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %i8, i8* align 4 bitcast (%struct.bitfield* @bitfield to i8*), i64 4, i1 false)
  ret void
; CHECK-LABEL: @test_bitfield(
; CHECK:       call void @llvm.memset
; CHECK-NOT:   call void @llvm.memcpy
; CHECK:       ret void
}

@i1x16_zero = internal constant <16 x i1> <i1 0, i1 0, i1 0, i1 0, i1 0, i1 0, i1 0, i1 0, i1 0, i1 0, i1 0, i1 0, i1 0, i1 0, i1 0, i1 0>, align 4
define void @test_i1x16_zero() nounwind {
  %a = alloca <16 x i1>, align 4
  %i8 = bitcast <16 x i1>* %a to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %i8, i8* align 4 bitcast (<16 x i1>* @i1x16_zero to i8*), i64 16, i1 false)
  ret void
; CHECK-LABEL: @test_i1x16_zero(
; CHECK:       call void @llvm.memset
; CHECK-NOT:   call void @llvm.memcpy
; CHECK:       ret void
}

; i1 isn't currently handled. Should it?
@i1x16_one = internal constant <16 x i1> <i1 1, i1 1, i1 1, i1 1, i1 1, i1 1, i1 1, i1 1, i1 1, i1 1, i1 1, i1 1, i1 1, i1 1, i1 1, i1 1>, align 4
define void @test_i1x16_one() nounwind {
  %a = alloca <16 x i1>, align 4
  %i8 = bitcast <16 x i1>* %a to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %i8, i8* align 4 bitcast (<16 x i1>* @i1x16_one to i8*), i64 16, i1 false)
  ret void
; CHECK-LABEL: @test_i1x16_one(
; CHECK-NOT:   call void @llvm.memset
; CHECK:      call void @llvm.memcpy
; CHECK:       ret void
}

@half = internal constant half 0xH0000, align 4
define void @test_half() nounwind {
  %a = alloca half, align 4
  %i8 = bitcast half* %a to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %i8, i8* align 4 bitcast (half* @half to i8*), i64 2, i1 false)
  ret void
; CHECK-LABEL: @test_half(
; CHECK:       call void @llvm.memset
; CHECK-NOT:   call void @llvm.memcpy
; CHECK:       ret void
}
