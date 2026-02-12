; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv-pc-vulkan1.3-library %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-pc-vulkan1.3-library %s -o - -filetype=obj | spirv-val %}

%struct.RuntimeArray = type { [0 x i32] }

@buffer = external addrspace(11) global %struct.RuntimeArray

; CHECK-DAG: %[[#int:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#ptr_sb_int:]] = OpTypePointer StorageBuffer %[[#int]]
; CHECK-DAG: %[[#rt_array:]] = OpTypeRuntimeArray %[[#int]]
; CHECK-DAG: %[[#struct:]] = OpTypeStruct %[[#rt_array]]
; CHECK-DAG: %[[#ptr_sb_struct:]] = OpTypePointer StorageBuffer %[[#struct]]
; CHECK-DAG: %[[#buffer:]] = OpVariable %[[#ptr_sb_struct]] StorageBuffer

define spir_func i32 @runtime_array_access(i32 %idx) convergent {
entry:
  %0 = call token @llvm.experimental.convergence.entry()

  %1 = call ptr addrspace(11) (ptr addrspace(11), ...) @llvm.structured.gep.p11(ptr addrspace(11) elementtype(%struct.RuntimeArray) @buffer, i32 0, i32 %idx)
  ; CHECK: %[[#ptr_elem:]] = OpInBoundsAccessChain %[[#ptr_sb_int]] %[[#buffer]] %[[#]] %[[#]]

  %2 = load i32, ptr addrspace(11) %1, align 4
  ; CHECK: %[[#val:]] = OpLoad %[[#int]] %[[#ptr_elem]] Aligned 4

  ret i32 %2
}

declare token @llvm.experimental.convergence.entry() #1
declare ptr addrspace(11) @llvm.structured.gep.p11(ptr addrspace(11), ...) #3

attributes #1 = { convergent nocallback nofree nosync nounwind willreturn memory(none) }
attributes #3 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
