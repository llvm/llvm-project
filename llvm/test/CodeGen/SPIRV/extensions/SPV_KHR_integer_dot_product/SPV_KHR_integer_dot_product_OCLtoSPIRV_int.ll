; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv32v1.6-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32v1.6-unknown-unknown %s -o - -filetype=obj | spirv-val %}
; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown --spirv-ext=+SPV_KHR_integer_dot_product %s -o - | FileCheck %s --check-prefixes=CHECK,CHECK-EXT
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown --spirv-ext=+SPV_KHR_integer_dot_product %s -o - -filetype=obj | spirv-val %}

; CHECK: Capability DotProduct
; CHECK: Capability DotProductInput4x8BitPacked
; CHECK-EXT: OpExtension "SPV_KHR_integer_dot_product"
; CHECK-NOT: OpExtension "SPV_KHR_integer_dot_product"

; CHECK: Name %[[#SignedA:]] "ia"
; CHECK: Name %[[#UnsignedA:]] "ua"
; CHECK: Name %[[#SignedB:]] "ib"
; CHECK: Name %[[#UnsignedB:]] "ub"

; CHECK: SDot %[[#]] %[[#SignedA]] %[[#SignedB]] 0
; CHECK: SUDot %[[#]] %[[#SignedA]] %[[#UnsignedB]] 0
; CHECK: SUDot %[[#]] %[[#SignedB]] %[[#UnsignedA]] 0
; CHECK: UDot %[[#]] %[[#UnsignedA]] %[[#UnsignedB]] 0

; CHECK: SDotAccSat %[[#]] %[[#SignedA]] %[[#SignedB]] %[[#]] 0
; CHECK: SUDotAccSat %[[#]] %[[#SignedA]] %[[#UnsignedB]] %[[#]] 0
; CHECK: SUDotAccSat %[[#]] %[[#SignedB]] %[[#UnsignedA]] %[[#]] 0
; CHECK: UDotAccSat %[[#]] %[[#UnsignedA]] %[[#UnsignedB]] %[[#]] 0

define spir_kernel void @test(i32 %ia, i32 %ua, i32 %ib, i32 %ub, i32 %ires, i32 %ures) {
entry:
  %call = tail call spir_func i32 @_Z20dot_4x8packed_ss_intjj(i32 %ia, i32 %ib) #2
  %call1 = tail call spir_func i32 @_Z20dot_4x8packed_su_intjj(i32 %ia, i32 %ub) #2
  %call2 = tail call spir_func i32 @_Z20dot_4x8packed_us_intjj(i32 %ua, i32 %ib) #2
  %call3 = tail call spir_func i32 @_Z21dot_4x8packed_uu_uintjj(i32 %ua, i32 %ub) #2
  %call4 = tail call spir_func i32 @_Z28dot_acc_sat_4x8packed_ss_intjji(i32 %ia, i32 %ib, i32 %ires) #2
  %call5 = tail call spir_func i32 @_Z28dot_acc_sat_4x8packed_su_intjji(i32 %ia, i32 %ub, i32 %ires) #2
  %call6 = tail call spir_func i32 @_Z28dot_acc_sat_4x8packed_us_intjji(i32 %ua, i32 %ib, i32 %ires) #2
  %call7 = tail call spir_func i32 @_Z29dot_acc_sat_4x8packed_uu_uintjjj(i32 %ua, i32 %ub, i32 %ures) #2
  ret void
}

declare spir_func i32 @_Z20dot_4x8packed_ss_intjj(i32, i32)
declare spir_func i32 @_Z20dot_4x8packed_su_intjj(i32, i32)
declare spir_func i32 @_Z20dot_4x8packed_us_intjj(i32, i32)
declare spir_func i32 @_Z21dot_4x8packed_uu_uintjj(i32, i32)
declare spir_func i32 @_Z28dot_acc_sat_4x8packed_ss_intjji(i32, i32, i32)
declare spir_func i32 @_Z28dot_acc_sat_4x8packed_su_intjji(i32, i32, i32)
declare spir_func i32 @_Z28dot_acc_sat_4x8packed_us_intjji(i32, i32, i32)
declare spir_func i32 @_Z29dot_acc_sat_4x8packed_uu_uintjjj(i32, i32, i32)

!llvm.module.flags = !{!0}
!opencl.ocl.version = !{!1}
!opencl.spir.version = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 2, i32 0}
