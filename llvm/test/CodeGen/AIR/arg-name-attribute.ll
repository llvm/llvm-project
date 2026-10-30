; A kernel parameter's air.arg_name is its "air-arg-name" attribute, which
; survives where the context discards value names (as release builds do):
; the globals buffer and the atomics lock table are told apart by it.
; RUN: llvm-as %s -o %t.bc
; RUN: llc -mtriple=air -discard-value-names %t.bc -o - | FileCheck %s

; CHECK-DAG: !"air.arg_name", !"zluda_globals"}
; CHECK-DAG: !"air.location_index", i32 30, {{.*}}!"air.arg_name", !"air_atomic_locks"}

define void @kernel(ptr addrspace(1) "air-arg-name"="zluda_globals" %g, ptr addrspace(1) %p) "air-kernel" {
entry:
  %old = atomicrmw fadd ptr addrspace(1) %p, <4 x float> <float 1.0, float 1.0, float 1.0, float 1.0> monotonic
  store i32 1, ptr addrspace(1) %g
  ret void
}
