; RUN: not opt -S -dxil-op-lower \
; RUN:   -mtriple=dxil-pc-shadermodel6.4-library %s 2>&1 | \
; RUN:   FileCheck %s --check-prefix=VERSION
; RUN: not opt -S -dxil-op-lower \
; RUN:   -mtriple=dxil-pc-shadermodel6.5-library %s 2>&1 | \
; RUN:   FileCheck %s --check-prefix=TYPE

; VERSION: Cannot create WaveMultiPrefixOp operation: No valid overloads for DXIL version 1.4
; TYPE: Cannot create WaveMultiPrefixOp operation: Invalid overload type
define i1 @test_bool(i1 %value) {
  %ret = call i1 @llvm.dx.wave.multi.prefix.sum.i1(
      i1 %value, i32 1, i32 2, i32 4, i32 8)
  ret i1 %ret
}
