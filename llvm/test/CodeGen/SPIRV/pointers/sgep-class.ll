; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv-pc-vulkan1.3-library %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-pc-vulkan1.3-library %s -o - -filetype=obj | spirv-val %}

%class.Base = type { i32 }
%class.Derived = type { %class.Base, float }

; CHECK-DAG: %[[#int:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#float:]] = OpTypeFloat 32
; CHECK-DAG: %[[#Base:]] = OpTypeStruct %[[#int]]
; CHECK-DAG: %[[#Derived:]] = OpTypeStruct %[[#Base]] %[[#float]]
; CHECK-DAG: %[[#ptr_Derived:]] = OpTypePointer Function %[[#Derived]]
; CHECK-DAG: %[[#ptr_Base:]] = OpTypePointer Function %[[#Base]]
; CHECK-DAG: %[[#ptr_int:]] = OpTypePointer Function %[[#int]]
; CHECK-DAG: %[[#idx_0:]] = OpConstant %[[#int]] 0

define spir_func void @class_access(ptr %d) convergent {
entry:
  %0 = call token @llvm.experimental.convergence.entry()
  ; CHECK: %[[#d_var:]] = OpFunctionParameter %[[#ptr_Derived]]

  ; Access Base part
  %1 = call ptr (ptr, ...) @llvm.structured.gep.p0(ptr elementtype(%class.Derived) %d, i32 0)
  ; CHECK: %[[#ptr_base:]] = OpInBoundsAccessChain %[[#ptr_Base]] %[[#d_var]] %[[#idx_0]]

  ; Access field in Base
  %2 = call ptr (ptr, ...) @llvm.structured.gep.p0(ptr elementtype(%class.Base) %1, i32 0)
  ; CHECK: %[[#ptr_field:]] = OpInBoundsAccessChain %[[#ptr_int]] %[[#ptr_base]] %[[#idx_0]]

  store i32 42, ptr %2, align 4
  ; CHECK: OpStore %[[#ptr_field]]

  ret void
}

declare token @llvm.experimental.convergence.entry() #1
declare ptr @llvm.structured.gep.p0(ptr, ...) #3

attributes #1 = { convergent nocallback nofree nosync nounwind willreturn memory(none) }
attributes #3 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
