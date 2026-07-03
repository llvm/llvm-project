; RUN: opt -passes=openmp-opt-cgscc -pass-remarks-analysis=openmp-opt -openmp-print-gpu-kernels -disable-output < %s 2>&1 | FileCheck %s --implicit-check-not=non_kernel

; CHECK-DAG: remark: <unknown>:0:0: OpenMP GPU kernel kernel1
; CHECK-DAG: remark: <unknown>:0:0: OpenMP GPU kernel kernel2

define ptx_kernel void @kernel1() "kernel" {
  ret void
}

define ptx_kernel void @kernel2() "kernel" {
  ret void
}

define void @non_kernel() {
  ret void
}

; Needed to trigger the openmp-opt pass
declare dso_local void @__kmpc_kernel_prepare_parallel(ptr)

!llvm.module.flags = !{!4}

!4 = !{i32 7, !"openmp", i32 50}
