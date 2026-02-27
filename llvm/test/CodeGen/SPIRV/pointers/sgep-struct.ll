; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv-pc-vulkan1.3-library %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-pc-vulkan1.3-library %s -o - -filetype=obj | spirv-val %}

%struct.Simple = type { i32, float, i64 }

; CHECK-DAG: %[[#int:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#float:]] = OpTypeFloat 32
; CHECK-DAG: %[[#long:]] = OpTypeInt 64 0
; CHECK-DAG: %[[#struct:]] = OpTypeStruct %[[#int]] %[[#float]] %[[#long]]
; CHECK-DAG: %[[#ptr_struct:]] = OpTypePointer Function %[[#struct]]
; CHECK-DAG: %[[#ptr_float:]] = OpTypePointer Function %[[#float]]
; CHECK-DAG: %[[#idx_1:]] = OpConstant %[[#int]] 1

define spir_func void @struct_access(ptr %s, ptr %out) convergent {
entry:
  %0 = call token @llvm.experimental.convergence.entry()
  ; CHECK: %[[#s_var:]] = OpFunctionParameter %[[#ptr_struct]]
  ; CHECK: %[[#out_var:]] = OpFunctionParameter %[[#ptr_float]]

  %1 = call ptr (ptr, ...) @llvm.structured.gep.p0(ptr elementtype(%struct.Simple) %s, i32 1)
  ; CHECK: %[[#ptr_field:]] = OpInBoundsAccessChain %[[#ptr_float]] %[[#s_var]] %[[#idx_1]]

  %2 = load float, ptr %1, align 4
  ; CHECK: %[[#val:]] = OpLoad %[[#float]] %[[#ptr_field]] Aligned 4

  store float %2, ptr %out, align 4
  ; CHECK: OpStore %[[#out_var]] %[[#val]] Aligned 4

  ret void
}

declare token @llvm.experimental.convergence.entry() #1
declare ptr @llvm.structured.gep.p0(ptr, ...) #3

attributes #1 = { convergent nocallback nofree nosync nounwind willreturn memory(none) }
attributes #3 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
