; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV

; CHECK-SPIRV-DAG: OpDecorate %[[Id:[0-9]+]] BuiltIn GlobalInvocationId
; CHECK-SPIRV-DAG: OpDecorate %[[Id:[0-9]+]] BuiltIn GlobalLinearId
; CHECK-SPIRV: %[[Id:[0-9]+]] = OpVariable %{{[0-9]+}}
; CHECK-SPIRV: %[[Id:[0-9]+]] = OpVariable %{{[0-9]+}}

define spir_kernel void @f(){
entry:
  %0 = call spir_func i32 @_Z29__spirv_BuiltInGlobalLinearIdv()
  %1 = call spir_func i64 @_Z33__spirv_BuiltInGlobalInvocationIdi(i32 1)
  ret void
}

declare spir_func i32 @_Z29__spirv_BuiltInGlobalLinearIdv()
declare spir_func i64 @_Z33__spirv_BuiltInGlobalInvocationIdi(i32)
