; RUN: opt -S -dxil-op-lower < %s | FileCheck %s

; Make sure dxil operation function calls for umax are generated for i32/i64.

target datalayout = "e-m:e-p:32:32-i1:32-i8:8-i16:16-i32:32-i64:64-f16:16-f32:32-f64:64-n8:16:32:64"
target triple = "dxil-pc-shadermodel6.7-library"

; CHECK-LABEL:test_umax_i32
; Function Attrs: noinline nounwind optnone
define noundef i32 @test_umax_i32(i32 noundef %a, i32 noundef %b) #0 {
entry:
; CHECK:call i32 @dx.op.binary.i32(i32 39, i32 %{{.*}}, i32 %{{.*}})
  %0 = call i32 @llvm.umax.i32(i32 %a, i32 %b)
  ret i32 %0
}

; CHECK-LABEL:test_umax_i64
define noundef i64 @test_umax_i64(i64 noundef %a, i64 noundef %b) #0 {
entry:
; CHECK:call i64 @dx.op.binary.i64(i32 39, i64 %{{.*}}, i64 %{{.*}})
  %0 = call i64 @llvm.umax.i64(i64 %a, i64 %b)
  ret i64 %0
}

; Function Attrs: nocallback nofree nosync nounwind readnone speculatable willreturn
declare i32 @llvm.umax.i32(i32, i32) #1
declare i64 @llvm.umax.i64(i64, i64) #1

attributes #0 = { noinline nounwind }
attributes #1 = { nocallback nofree nosync nounwind readnone speculatable willreturn }
