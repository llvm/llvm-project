; RUN: opt -S -passes=amdgpu-unused-libfunc-cleanup -mtriple=amdgcn-- < %s | FileCheck %s

; Test that AMDGPUUnusedLibFuncCleanupPass keeps __ocml_sincos_f64 when it
; has callers — the optimisation pass successfully merged sin+cos into sincos.

declare float @__ocml_sincos_f32(float, ptr addrspace(5) writeonly)
declare double @__ocml_sincos_f64(double, ptr addrspace(5) writeonly)

@llvm.compiler.used = appending global [2 x ptr] [
  ptr @__ocml_sincos_f32,
  ptr @__ocml_sincos_f64
], section "llvm.metadata"

; __ocml_sincos_f64 is called — it should be kept.
; __ocml_sincos_f32 is not called — it should be removed.

; CHECK: @llvm.compiler.used = appending {{.*}}global [1 x ptr] [ptr @__ocml_sincos_f64], section "llvm.metadata"
; CHECK-NOT: declare float @__ocml_sincos_f32
; CHECK: declare double @__ocml_sincos_f64(double, ptr addrspace(5) writeonly)

define void @kernel(double %x, ptr addrspace(1) %out) {
  %tmp = alloca double, addrspace(5)
  %sin = call double @__ocml_sincos_f64(double %x, ptr addrspace(5) %tmp)
  %cos = load double, ptr addrspace(5) %tmp
  %sum = fadd double %sin, %cos
  store double %sum, ptr addrspace(1) %out
  ret void
}
