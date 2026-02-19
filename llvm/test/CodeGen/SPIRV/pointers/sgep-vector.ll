; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv-pc-vulkan1.3-library %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-pc-vulkan1.3-library %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: %[[#float:]] = OpTypeFloat 32
; CHECK-DAG: %[[#vec4:]] = OpTypeVector %[[#float]] 4
; CHECK-DAG: %[[#ptr_vec4:]] = OpTypePointer Function %[[#vec4]]
; CHECK-DAG: %[[#ptr_float:]] = OpTypePointer Function %[[#float]]
; CHECK-DAG: %[[#int:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#idx_0:]] = OpConstant %[[#int]] 0
; CHECK-DAG: %[[#idx_1:]] = OpConstant %[[#int]] 1
; CHECK-DAG: %[[#idx_2:]] = OpConstant %[[#int]] 2
; CHECK-DAG: %[[#idx_5:]] = OpConstant %[[#int]] 5
; CHECK-DAG: %[[#idx_10:]] = OpConstant %[[#int]] 10
; CHECK-DAG: %[[#array_vec4:]] = OpTypeArray %[[#vec4]] %[[#idx_2]]
; CHECK-DAG: %[[#ptr_array_vec4:]] = OpTypePointer Function %[[#array_vec4]]
; CHECK-DAG: %[[#array_float:]] = OpTypeArray %[[#float]] %[[#idx_10]]
; CHECK-DAG: %[[#ptr_array_float:]] = OpTypePointer Function %[[#array_float]]

define spir_func void @vector_access(ptr %vec, ptr %out) convergent {
entry:
  %0 = call token @llvm.experimental.convergence.entry()
  ; CHECK: %[[#vec_var:]] = OpFunctionParameter %[[#ptr_vec4]]
  ; CHECK: %[[#out_var:]] = OpFunctionParameter %[[#ptr_float]]

  %1 = call ptr (ptr, ...) @llvm.structured.gep.p0(ptr elementtype(<4 x float>) %vec, i32 2)
  ; CHECK: %[[#ptr_elem:]] = OpInBoundsAccessChain %[[#ptr_float]] %[[#vec_var]] %[[#idx_2]]

  %2 = load float, ptr %1, align 4
  ; CHECK: %[[#val:]] = OpLoad %[[#float]] %[[#ptr_elem]] Aligned 4

  store float %2, ptr %out, align 4
  ; CHECK: OpStore %[[#out_var]] %[[#val]] Aligned 4

  ret void
}

define spir_func void @array_of_vectors_access(ptr %arr, ptr %out) convergent {
entry:
  %0 = call token @llvm.experimental.convergence.entry()
  ; CHECK: %[[#arr_var:]] = OpFunctionParameter %[[#ptr_array_vec4]]
  ; CHECK: %[[#out_var2:]] = OpFunctionParameter %[[#ptr_float]]

  %1 = call ptr (ptr, ...) @llvm.structured.gep.p0(ptr elementtype([2 x <4 x float>]) %arr, i32 1, i32 2)
  ; CHECK: %[[#ptr_elem2:]] = OpInBoundsAccessChain %[[#ptr_float]] %[[#arr_var]] %[[#idx_1]] %[[#idx_2]]

  %2 = load float, ptr %1, align 4
  ; CHECK: %[[#val2:]] = OpLoad %[[#float]] %[[#ptr_elem2]] Aligned 4

  store float %2, ptr %out, align 4
  ; CHECK: OpStore %[[#out_var2]] %[[#val2]] Aligned 4

  ret void
}

define spir_func void @array_access(ptr %arr, ptr %out) convergent {
entry:
  %0 = call token @llvm.experimental.convergence.entry()
  ; CHECK: %[[#arr_var3:]] = OpFunctionParameter %[[#ptr_array_float]]
  ; CHECK: %[[#out_var3:]] = OpFunctionParameter %[[#ptr_float]]

  %1 = call ptr (ptr, ...) @llvm.structured.gep.p0(ptr elementtype([10 x float]) %arr, i32 0)
  ; CHECK: %[[#ptr_elem3_0:]] = OpInBoundsAccessChain %[[#ptr_float]] %[[#arr_var3]] %[[#idx_0]]

  %2 = load float, ptr %1, align 4
  ; CHECK: %[[#val3_0:]] = OpLoad %[[#float]] %[[#ptr_elem3_0]] Aligned 4

  %3 = call ptr (ptr, ...) @llvm.structured.gep.p0(ptr elementtype([10 x float]) %arr, i32 5)
  ; CHECK: %[[#ptr_elem3_5:]] = OpInBoundsAccessChain %[[#ptr_float]] %[[#arr_var3]] %[[#idx_5]]

  %4 = load float, ptr %3, align 4
  ; CHECK: %[[#val3_5:]] = OpLoad %[[#float]] %[[#ptr_elem3_5]] Aligned 4

  %5 = fadd float %2, %4
  ; CHECK: %[[#res:]] = OpFAdd %[[#float]] %[[#val3_0]] %[[#val3_5]]

  store float %5, ptr %out, align 4
  ; CHECK: OpStore %[[#out_var3]] %[[#res]] Aligned 4

  ret void
}

declare token @llvm.experimental.convergence.entry() #1
declare ptr @llvm.structured.gep.p0(ptr, ...) #3

attributes #1 = { convergent nocallback nofree nosync nounwind willreturn memory(none) }
attributes #3 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
