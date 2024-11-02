; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV

; CHECK-SPIRV: OpDecorate %[[#Id:]] BuiltIn GlobalLinearId
; CHECK-SPIRV: %[[#Id:]] = OpVariable %[[#]]

@__spirv_BuiltInGlobalLinearId = external addrspace(1) global i32

define spir_kernel void @f(i32 addrspace(1)* nocapture %order) {
entry:
  %0 = load i32, i32 addrspace(4)* addrspacecast (i32 addrspace(1)* @__spirv_BuiltInGlobalLinearId to i32 addrspace(4)*), align 4
  ;; Need to store the result somewhere, otherwise the access to GlobalLinearId
  ;; may be removed.
  store i32 %0, i32 addrspace(1)* %order, align 4
  ret void
}
