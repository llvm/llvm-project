; RUN: opt -passes=openmpopt -pass-remarks=openmp-opt -openmp-print-gpu-kernels -disable-output < %s 2>&1 | FileCheck %s --implicit-check-not=non_kernel
; RUN: opt        -openmpopt -pass-remarks=openmp-opt -openmp-print-gpu-kernels -disable-output < %s 2>&1 | FileCheck %s --implicit-check-not=non_kernel

; CHECK-DAG: remark: <unknown>:0:0: OpenMP GPU kernel kernel1
; CHECK-DAG: remark: <unknown>:0:0: OpenMP GPU kernel kernel2

define void @kernel1() {
  ret void
}

define void @kernel2() {
  ret void
}

define void @non_kernel() {
  ret void
}

; Needed to trigger the openmp-opt pass
declare dso_local void @__kmpc_kernel_prepare_parallel(i8*)

!nvvm.annotations = !{!2, !0, !1, !3, !1, !2}

!0 = !{void ()* @kernel1, !"kernel", i32 1}
!1 = !{void ()* @non_kernel, !"non_kernel", i32 1}
!2 = !{null, !"align", i32 1}
!3 = !{void ()* @kernel2, !"kernel", i32 1}
