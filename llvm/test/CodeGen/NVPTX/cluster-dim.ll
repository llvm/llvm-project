; RUN: llc < %s -march=nvptx64 -mcpu=sm_90 | FileCheck %s
; RUN: %if ptxas-12.0 %{ llc < %s -march=nvptx64 -mcpu=sm_90 | %ptxas-verify -arch=sm_90 %}

; CHECK-LABEL: .entry kernel_func_clusterxyz
define void @kernel_func_clusterxyz() {
; CHECK: .explicitcluster
; CHECK: .reqnctapercluster 3, 5, 7
  ret void
}


!nvvm.annotations = !{!1, !2}

!1 = !{ptr @kernel_func_clusterxyz, !"kernel", i32 1}
!2 = !{ptr @kernel_func_clusterxyz, !"cluster_dim_x", i32 3, !"cluster_dim_y", i32 5, !"cluster_dim_z", i32 7}
