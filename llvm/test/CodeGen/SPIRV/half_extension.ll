;; #pragma OPENCL EXTENSION cl_khr_fp16 : enable
;; half test()
;; {
;;   half x = 0.1f;
;;   x += 2.0f;
;;   half y = x + x;
;;   return y;
;; }

; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV

; CHECK-SPIRV-DAG: OpCapability Float16Buffer
; CHECK-SPIRV-DAG: OpCapability Float16

define spir_func half @test() {
entry:
  %x = alloca half, align 2
  %y = alloca half, align 2
  store half 0xH2E66, half* %x, align 2
  %0 = load half, half* %x, align 2
  %conv = fpext half %0 to float
  %add = fadd float %conv, 2.000000e+00
  %conv1 = fptrunc float %add to half
  store half %conv1, half* %x, align 2
  %1 = load half, half* %x, align 2
  %2 = load half, half* %x, align 2
  %add2 = fadd half %1, %2
  store half %add2, half* %y, align 2
  %3 = load half, half* %y, align 2
  ret half %3
}
