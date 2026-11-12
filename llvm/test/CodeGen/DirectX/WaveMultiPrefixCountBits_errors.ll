; RUN: not opt -S -dxil-op-lower \
; RUN:   -mtriple=dxil-pc-shadermodel6.4-library < %s 2>&1 | FileCheck %s

; CHECK: Cannot create WaveMultiPrefixBitCount operation: No valid stage for DXIL version 1.4
define i32 @test_version(i1 %value) {
  %ret = call i32 @llvm.dx.wave.multi.prefix.bit.count(
      i1 %value, i32 1, i32 2, i32 4, i32 8)
  ret i32 %ret
}
