; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV

; CHECK-SPIRV-DAG: OpEntryPoint Kernel %[[#WORKER:]] "worker"
; CHECK-SPIRV-DAG: OpExecutionMode %[[#WORKER]] LocalSizeHint 128 10 1

define spir_kernel void @worker() local_unnamed_addr !work_group_size_hint !3 {
entry:
  ret void
}

!3 = !{i32 128, i32 10, i32 1}
