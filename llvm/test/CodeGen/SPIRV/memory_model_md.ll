; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s --check-prefix=SPV
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32v1.2-unknown-unknown %s -o - -filetype=obj | spirv-val --target-env opencl2.2 %}

; SPV: OpMemoryModel Physical32 OpenCL
define dso_local dllexport void @k_no_fc(i32 %ibuf, i32 %obuf) local_unnamed_addr {
entry:
  ret void
}

!spirv.MemoryModel = !{!0}

!0 = !{i32 1, i32 2}
