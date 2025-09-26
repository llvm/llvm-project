; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: OpName %[[#Fun:]] "_Z3booi"
; CHECK-DAG: OpDecorate %[[#Param:]] FuncParamAttr Sret
; CHECK-DAG: %[[#PtrTy:]] = OpTypePointer Function %[[#StructTy:]]
; CHECK-DAG: %[[#StructTy]] = OpTypeStruct 
; CHECK: %[[#Fun]] = OpFunction %[[#]] 
; CHECK: %[[#Param]] = OpFunctionParameter %[[#PtrTy]] 

%struct.Example = type { }

define spir_func i32 @foo() {
  %1 = alloca %struct.Example, align 8
  call void @_Z3booi(ptr sret(%struct.Example) align 8 %1, i32 noundef 42)
  ret i32 0
}

declare void @_Z3booi(ptr sret(%struct.Example) align 8, i32 noundef)
