; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV

; CHECK-SPIRV: OpCapability Groups
; CHECK-SPIRV: %[[#BoolTypeID:]] = OpTypeBool
; CHECK-SPIRV: %[[#ConstID:]] = OpConstantTrue %[[#BoolTypeID]]
; CHECK-SPIRV: %[[#]] = OpGroupAll %[[#BoolTypeID]] %[[#]] %[[#ConstID]]
; CHECK-SPIRV: %[[#]] = OpGroupAny %[[#BoolTypeID]] %[[#]] %[[#ConstID]]

define spir_kernel void @test(i32 addrspace(1)* nocapture readnone %i) {
entry:
  %call = tail call spir_func i32 @_Z14work_group_alli(i32 5)
  %call1 = tail call spir_func i32 @_Z14work_group_anyi(i32 5)
  ret void
}

declare spir_func i32 @_Z14work_group_alli(i32)

declare spir_func i32 @_Z14work_group_anyi(i32)
