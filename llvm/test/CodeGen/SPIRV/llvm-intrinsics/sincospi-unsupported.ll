; RUN: split-file %s %t
; RUN: not llc -O0 -mtriple=spirv-unknown-vulkan1.3-compute %t/f32.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=SHADER
; RUN: not llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_EXT_long_vector %t/v5.ll -o /dev/null 2>&1 | FileCheck %s
; RUN: not llc -O0 -mtriple=spirv64-unknown-unknown %t/f80.ll -o /dev/null 2>&1 | FileCheck %s

; SHADER: llvm.sincospi requires the OpenCL.std extended instruction set
; CHECK: llvm.sincospi requires a 16, 32 or 64-bit float or a vector of 2, 3, 4, 8 or 16

;--- f32.ll
define float @f32(float %x) {
  %r = call { float, float } @llvm.sincospi.f32(float %x)
  %sin = extractvalue { float, float } %r, 0
  ret float %sin
}

;--- v5.ll
define <5 x float> @v5(<5 x float> %x) {
  %r = call { <5 x float>, <5 x float> } @llvm.sincospi.v5f32(<5 x float> %x)
  %sin = extractvalue { <5 x float>, <5 x float> } %r, 0
  ret <5 x float> %sin
}

;--- f80.ll
define x86_fp80 @f80(x86_fp80 %x) {
  %r = call { x86_fp80, x86_fp80 } @llvm.sincospi.f80(x86_fp80 %x)
  %sin = extractvalue { x86_fp80, x86_fp80 } %r, 0
  ret x86_fp80 %sin
}
