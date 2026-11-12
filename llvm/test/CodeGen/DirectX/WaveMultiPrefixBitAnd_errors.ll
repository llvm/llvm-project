; RUN: not opt -S -dxil-op-lower \
; RUN:   -mtriple=dxil-pc-shadermodel6.4-library %s 2>&1 | \
; RUN:   FileCheck %s --check-prefix=VERSION
; RUN: not opt -S -dxil-op-lower \
; RUN:   -mtriple=dxil-pc-shadermodel6.5-library %s 2>&1 | \
; RUN:   FileCheck %s --check-prefix=TYPE

; VERSION: Cannot create WaveMultiPrefixOp operation: No valid overloads for DXIL version 1.4
; TYPE: Cannot create WaveMultiPrefixOp operation: Invalid overload type
define i8 @test_i8(i8 %value) {
  %ret = call i8 @llvm.dx.wave.multi.prefix.bit.and.i8(
      i8 %value, i32 1, i32 2, i32 4, i32 8)
  ret i8 %ret
}
