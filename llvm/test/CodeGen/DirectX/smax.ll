; RUN: opt -S -dxil-op-lower -mtriple=dxil-pc-shadermodel6.3-library %s | FileCheck %s

; Make sure dxil operation function calls for smax are generated for i16/i32/i64.

; CHECK-LABEL:test_smax_i16
define noundef i16 @test_smax_i16(i16 noundef %a, i16 noundef %b) {
entry:
; CHECK: call i16 @dx.op.binary.i16(i32 37, i16 %{{.*}}, i16 %{{.*}}) #[[#ATTR:]]
  %0 = call i16 @llvm.smax.i16(i16 %a, i16 %b)
  ret i16 %0
}

; CHECK-LABEL:test_smax_i32
define noundef i32 @test_smax_i32(i32 noundef %a, i32 noundef %b) {
entry:
; CHECK: call i32 @dx.op.binary.i32(i32 37, i32 %{{.*}}, i32 %{{.*}}) #[[#ATTR]]
  %0 = call i32 @llvm.smax.i32(i32 %a, i32 %b)
  ret i32 %0
}

; CHECK-LABEL:test_smax_i64
define noundef i64 @test_smax_i64(i64 noundef %a, i64 noundef %b) {
entry:
; CHECK: call i64 @dx.op.binary.i64(i32 37, i64 %{{.*}}, i64 %{{.*}}) #[[#ATTR]]
  %0 = call i64 @llvm.smax.i64(i64 %a, i64 %b)
  ret i64 %0
}

; CHECK: attributes #[[#ATTR]] = {{{.*}} memory(none) {{.*}}}

declare i16 @llvm.smax.i16(i16, i16)
declare i32 @llvm.smax.i32(i32, i32)
declare i64 @llvm.smax.i64(i64, i64)
