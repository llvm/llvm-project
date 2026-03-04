;; Make sure the backend doesn't crash if the input LLVM IR contains llvm.invariant.* intrinsics
; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-linux %s -o - | FileCheck %s

; CHECK-NOT: OpFunctionParameter
; CHECK-NOT: OpFunctionCall

@WGSharedVar = internal addrspace(3) constant i64 0, align 8

declare ptr @llvm.invariant.start.p3(i64 immarg, ptr addrspace(3) nocapture)

declare void @llvm.invariant.end.p3(ptr, i64 immarg, ptr addrspace(3) nocapture)

define linkonce_odr dso_local spir_func void @func() {
  store i64 2, ptr addrspace(3) @WGSharedVar
  %1 = bitcast ptr addrspace(3) @WGSharedVar to ptr addrspace(3)
  %2 = call ptr @llvm.invariant.start.p3(i64 8, ptr addrspace(3) %1)
  call void @llvm.invariant.end.p3(ptr %2, i64 8, ptr addrspace(3) %1)
  ret void
}
