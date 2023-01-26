;; Check translation of intel_reqd_sub_group_size metadata to SubgroupSize
;; execution mode and back. The IR is producded from the following OpenCL C code:
;; kernel __attribute__((intel_reqd_sub_group_size(8)))
;; void foo() {}

; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV

; CHECK-SPIRV: OpCapability SubgroupDispatch
; CHECK-SPIRV: OpEntryPoint Kernel %[[#kernel:]] "foo"
; CHECK-SPIRV: OpExecutionMode %[[#kernel]] SubgroupSize 8

define spir_kernel void @foo() !intel_reqd_sub_group_size !0 {
entry:
  ret void
}

!0 = !{i32 8}
