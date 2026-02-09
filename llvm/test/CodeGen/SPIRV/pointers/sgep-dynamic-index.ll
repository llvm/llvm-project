; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv-pc-vulkan1.3-library %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-pc-vulkan1.3-library %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: %[[#int:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#long:]] = OpTypeInt 64 0
; CHECK-DAG: %[[#ptr_int:]] = OpTypePointer Function %[[#int]]
; CHECK-DAG: %[[#idx_10:]] = OpConstant %[[#int]] 10
; CHECK-DAG: %[[#array:]] = OpTypeArray %[[#int]] %[[#idx_10]]
; CHECK-DAG: %[[#ptr_array:]] = OpTypePointer Function %[[#array]]

define spir_func i32 @dynamic_access(ptr %arr, i32 %idx) convergent {
entry:
  %0 = call token @llvm.experimental.convergence.entry()
  ; CHECK: %[[#arr_var:]] = OpFunctionParameter %[[#ptr_array]]
  ; CHECK: %[[#idx_var:]] = OpFunctionParameter %[[#int]]

  %1 = call ptr (ptr, ...) @llvm.structured.gep.p0(ptr elementtype([10 x i32]) %arr, i32 %idx)
  ; CHECK: %[[#ptr_elem:]] = OpInBoundsAccessChain %[[#ptr_int]] %[[#arr_var]] %[[#idx_var]]

  %2 = load i32, ptr %1, align 4
  ; CHECK: %[[#val:]] = OpLoad %[[#int]] %[[#ptr_elem]] Aligned 4

  ret i32 %2
  ; CHECK: OpReturnValue %[[#val]]
}

define spir_func i32 @dynamic_access_i64(ptr %arr, i64 %idx) convergent {
entry:
  %0 = call token @llvm.experimental.convergence.entry()
  ; CHECK: %[[#arr_var2:]] = OpFunctionParameter %[[#ptr_array]]
  ; CHECK: %[[#idx_var2:]] = OpFunctionParameter %[[#long]]

  %1 = call ptr (ptr, ...) @llvm.structured.gep.p0(ptr elementtype([10 x i32]) %arr, i64 %idx)
  ; CHECK: %[[#ptr_elem2:]] = OpInBoundsAccessChain %[[#ptr_int]] %[[#arr_var2]] %[[#idx_var2]]

  %2 = load i32, ptr %1, align 4
  ; CHECK: %[[#val2:]] = OpLoad %[[#int]] %[[#ptr_elem2]] Aligned 4

  ret i32 %2
  ; CHECK: OpReturnValue %[[#val2]]
}

declare token @llvm.experimental.convergence.entry() #1
declare ptr @llvm.structured.gep.p0(ptr, ...) #3

attributes #1 = { convergent nocallback nofree nosync nounwind willreturn memory(none) }
attributes #3 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
