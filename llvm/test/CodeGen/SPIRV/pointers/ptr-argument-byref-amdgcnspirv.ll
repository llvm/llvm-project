; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck --check-prefixes=CHECK,SPIRV %s
; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-amd-amdhsa %s -o - | FileCheck --check-prefixes=CHECK,AMDGCNSPIRV %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-amd-amdhsa %s -o - -filetype=obj | spirv-val %}

; CHECK: OpName %[[#XKER:]] "x"
; CHECK-DAG: OpName %[[#XFN:]] "x"
; SPIRV-NOT: OpDecorate %[[#XKER]] FuncParamAttr ByVal
; AMDGCNSPIRV: OpDecorate %[[#XKER]] FuncParamAttr ByVal
; SPIRV-NOT: OpDecorate %[[#XFN]] FuncParamAttr ByVal
; AMDGCNSPIRV: OpDecorate %[[#XFN]] FuncParamAttr ByVal

%struct.S = type { i32 }
%struct.SS = type { [7 x %struct.S] }

define spir_kernel void @ker(ptr addrspace(2) noundef byref(%struct.SS) %x) {
entry:
  ret void
}

define spir_func void @fn(ptr noundef byref(%struct.SS) %x) {
entry:
  ret void
}