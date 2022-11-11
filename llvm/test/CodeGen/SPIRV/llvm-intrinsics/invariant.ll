;; Make sure the backend doesn't crash if the input LLVM IR contains llvm.invariant.* intrinsics
; RUN: llc -O0 -mtriple=spirv64-unknown-linux %s -o - | FileCheck %s

; CHECK-NOT: OpFunctionParameter
; CHECK-NOT: OpFunctionCall

@WGSharedVar = internal addrspace(3) constant i64 0, align 8

declare {}* @llvm.invariant.start.p3i8(i64 immarg, i8 addrspace(3)* nocapture)

declare void @llvm.invariant.end.p3i8({}*, i64 immarg, i8 addrspace(3)* nocapture)

define linkonce_odr dso_local spir_func void @func() {
  store i64 2, i64 addrspace(3)* @WGSharedVar
  %1 = bitcast i64 addrspace(3)* @WGSharedVar to i8 addrspace(3)*
  %2 = call {}* @llvm.invariant.start.p3i8(i64 8, i8 addrspace(3)* %1)
  call void @llvm.invariant.end.p3i8({}* %2, i64 8, i8 addrspace(3)* %1)
  ret void
}
