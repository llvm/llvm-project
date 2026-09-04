; RUN: opt -S -scalarizer -dxil-op-lower -mtriple=dxil-pc-shadermodel6.3-library < %s | FileCheck %s

; Test that WavePrefixCountBits maps down to the DirectX op

define noundef i32 @wave_prefix_count_bits(i1 noundef %expr) {
entry:
; CHECK: call i32 @dx.op.wavePrefixOp(i32 136, i1 %expr)
  %ret = call i32 @llvm.dx.wave.prefix.bit.count(i1 %expr)
  ret i32 %ret
}
