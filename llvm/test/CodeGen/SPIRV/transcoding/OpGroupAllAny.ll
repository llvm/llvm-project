; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-SPIRV: OpCapability Groups
; CHECK-SPIRV-DAG: %[[#BoolTypeID:]] = OpTypeBool
; CHECK-SPIRV-DAG: %[[#True:]] = OpConstantTrue %[[#BoolTypeID]]
; CHECK-SPIRV-DAG: %[[#False:]] = OpConstantFalse %[[#BoolTypeID]]
; CHECK-SPIRV: %[[#]] = OpGroupAll %[[#BoolTypeID]] %[[#]] %[[#True]]
; CHECK-SPIRV: %[[#]] = OpGroupAny %[[#BoolTypeID]] %[[#]] %[[#True]]
; CHECK-SPIRV: %[[#]] = OpGroupAll %[[#BoolTypeID]] %[[#]] %[[#True]]
; CHECK-SPIRV: %[[#]] = OpGroupAny %[[#BoolTypeID]] %[[#]] %[[#False]]

define spir_kernel void @test(i32 addrspace(1)* nocapture readnone %i) {
entry:
  %call = tail call spir_func i32 @_Z14work_group_alli(i32 5)
  %call1 = tail call spir_func i32 @_Z14work_group_anyi(i32 5)
  %call3 = tail call spir_func i32 @__spirv_GroupAll(i32 0, i1 1)
  %call4 = tail call spir_func i32 @__spirv_GroupAny(i32 0, i1 0)
  ret void
}

declare spir_func i32 @_Z14work_group_alli(i32)
declare spir_func i32 @_Z14work_group_anyi(i32)

declare spir_func i1 @__spirv_GroupAll(i32, i1)
declare spir_func i1 @__spirv_GroupAny(i32, i1)
