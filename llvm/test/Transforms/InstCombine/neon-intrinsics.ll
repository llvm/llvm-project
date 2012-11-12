; RUN: opt < %s -instcombine -S | FileCheck %s

; The alignment arguments for NEON load/store intrinsics can be increased
; by instcombine.  Check for this.

; CHECK: vld4.v2i32({{.*}}, i32 32)
; CHECK: vst4.v2i32({{.*}}, i32 16)

@x = common global [8 x i32] zeroinitializer, align 32
@y = common global [8 x i32] zeroinitializer, align 16

%struct.__neon_int32x2x4_t = type { <2 x i32>, <2 x i32>, <2 x i32>, <2 x i32> }

define void @test() nounwind ssp {
  %tmp1 = call %struct.__neon_int32x2x4_t @llvm.arm.neon.vld4.v2i32(i8* bitcast ([8 x i32]* @x to i8*), i32 1)
  %tmp2 = extractvalue %struct.__neon_int32x2x4_t %tmp1, 0
  %tmp3 = extractvalue %struct.__neon_int32x2x4_t %tmp1, 1
  %tmp4 = extractvalue %struct.__neon_int32x2x4_t %tmp1, 2
  %tmp5 = extractvalue %struct.__neon_int32x2x4_t %tmp1, 3
  call void @llvm.arm.neon.vst4.v2i32(i8* bitcast ([8 x i32]* @y to i8*), <2 x i32> %tmp2, <2 x i32> %tmp3, <2 x i32> %tmp4, <2 x i32> %tmp5, i32 1)
  ret void
}

declare %struct.__neon_int32x2x4_t @llvm.arm.neon.vld4.v2i32(i8*, i32) nounwind readonly
declare void @llvm.arm.neon.vst4.v2i32(i8*, <2 x i32>, <2 x i32>, <2 x i32>, <2 x i32>, i32) nounwind
