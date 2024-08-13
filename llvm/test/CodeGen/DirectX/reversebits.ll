; RUN: opt -S -dxil-op-lower -mtriple=dxil-pc-shadermodel6.3-library %s | FileCheck %s

; Make sure dxil operation function calls for reversebits are generated for all integer types.

; Function Attrs: nounwind
define noundef i16 @test_bitreverse_short(i16 noundef %a) {
entry:
; CHECK:call i16 @dx.op.unary.i16(i32 30, i16 %{{.*}})
  %elt.bitreverse = call i16 @llvm.bitreverse.i16(i16 %a)
  ret i16 %elt.bitreverse
}

; Function Attrs: nounwind
define noundef i32 @test_bitreverse_int(i32 noundef %a) {
entry:
; CHECK:call i32 @dx.op.unary.i32(i32 30, i32 %{{.*}})
  %elt.bitreverse = call i32 @llvm.bitreverse.i32(i32 %a)
  ret i32 %elt.bitreverse
}

; Function Attrs: nounwind
define noundef i64 @test_bitreverse_long(i64 noundef %a) {
entry:
; CHECK:call i64 @dx.op.unary.i64(i32 30, i64 %{{.*}})
  %elt.bitreverse = call i64 @llvm.bitreverse.i64(i64 %a)
  ret i64 %elt.bitreverse
}

declare i16 @llvm.bitreverse.i16(i16)
declare i32 @llvm.bitreverse.i32(i32)
declare i64 @llvm.bitreverse.i64(i64)
