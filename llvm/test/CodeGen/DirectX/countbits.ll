; RUN: opt -S -dxil-op-lower -mtriple=dxil-pc-shadermodel6.3-library %s | FileCheck %s

; Make sure dxil operation function calls for countbits are generated for all integer types.

define noundef i16 @test_countbits_short(i16 noundef %a) {
entry:
; CHECK: call i16 @dx.op.unary.i16(i32 31, i16 %{{.*}})
  %elt.ctpop = call i16 @llvm.ctpop.i16(i16 %a)
  ret i16 %elt.ctpop
}

define noundef i32 @test_countbits_int(i32 noundef %a) {
entry:
; CHECK: call i32 @dx.op.unary.i32(i32 31, i32 %{{.*}})
  %elt.ctpop = call i32 @llvm.ctpop.i32(i32 %a)
  ret i32 %elt.ctpop
}

define noundef i64 @test_countbits_long(i64 noundef %a) {
entry:
; CHECK: call i64 @dx.op.unary.i64(i32 31, i64 %{{.*}})
  %elt.ctpop = call i64 @llvm.ctpop.i64(i64 %a)
  ret i64 %elt.ctpop
}

declare i16 @llvm.ctpop.i16(i16)
declare i32 @llvm.ctpop.i32(i32)
declare i64 @llvm.ctpop.i64(i64)
