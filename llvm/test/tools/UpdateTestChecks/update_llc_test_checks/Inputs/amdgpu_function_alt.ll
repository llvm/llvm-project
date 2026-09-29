; RUN: cat %S/amdgpu_function_alt.s | FileCheck --check-prefixes=CHECK %s

define float @sample(float %x) {
  %y = fmul float %x, %x
  ret float %y
}
