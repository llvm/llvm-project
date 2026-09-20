; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK: %[[#extinst_id:]] = OpExtInstImport "OpenCL.std"
; CHECK-DAG: %[[#i32:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#i64:]] = OpTypeInt 64 0
; CHECK-DAG: %[[#mask24:]] = OpConstant %[[#i32]] 16777215
; CHECK-DAG: %[[#mask40:]] = OpConstant %[[#i64]] 1099511627775
; CHECK-DAG: %[[#diff24:]] = OpConstant %[[#i32]] 8
; CHECK-DAG: %[[#diff40:]] = OpConstant %[[#i64]] 24

; CHECK: %[[#]] = OpFunction
; CHECK: %[[#]] = OpExtInst %[[#]] %[[#extinst_id]] clz
; CHECK: OpFunctionEnd

define spir_func i32 @TestClz(i32 %x) local_unnamed_addr {
entry:
  %0 = tail call i32 @llvm.ctlz.i32(i32 %x, i1 true)
  ret i32 %0
}

declare i32 @llvm.ctlz.i32(i32, i1 immarg)

; CHECK: %[[#]] = OpFunction
; CHECK: %[[#sum24:]] = OpIAdd %[[#i32]]
; CHECK: %[[#masked24:]] = OpBitwiseAnd %[[#i32]] %[[#sum24]] %[[#mask24]]
; CHECK: %[[#count24:]] = OpExtInst %[[#i32]] %[[#extinst_id]] clz %[[#masked24]]
; CHECK: %[[#result24:]] = OpISub %[[#i32]] %[[#count24]] %[[#diff24]]
; CHECK: OpReturnValue %[[#result24]]

define spir_func i24 @ctlz_i24(i32 %x) {
  %narrow = trunc i32 %x to i24
  %sum = add i24 %narrow, 1
  %count = call i24 @llvm.ctlz.i24(i24 %sum, i1 false)
  ret i24 %count
}

; CHECK: %[[#]] = OpFunction
; CHECK: %[[#shift24:]] = OpShiftLeftLogical %[[#i32]] %[[#]] %[[#diff24]]
; CHECK: %[[#poison24:]] = OpExtInst %[[#i32]] %[[#extinst_id]] clz %[[#shift24]]
; CHECK: OpReturnValue %[[#poison24]]

define spir_func i24 @ctlz_i24_zero_poison(i32 %x) {
  %narrow = trunc i32 %x to i24
  %count = call i24 @llvm.ctlz.i24(i24 %narrow, i1 true)
  ret i24 %count
}

; CHECK: %[[#]] = OpFunction
; CHECK: %[[#sum40:]] = OpIAdd %[[#i64]]
; CHECK: %[[#masked40:]] = OpBitwiseAnd %[[#i64]] %[[#sum40]] %[[#mask40]]
; CHECK: %[[#count40:]] = OpExtInst %[[#i64]] %[[#extinst_id]] clz %[[#masked40]]
; CHECK: %[[#result40:]] = OpISub %[[#i64]] %[[#count40]] %[[#diff40]]
; CHECK: OpReturnValue %[[#result40]]

define spir_func i40 @ctlz_i40(i64 %x) {
  %narrow = trunc i64 %x to i40
  %sum = add i40 %narrow, 1
  %count = call i40 @llvm.ctlz.i40(i40 %sum, i1 false)
  ret i40 %count
}

; The high bits of an i24 parameter are genuinely undefined, so the mask is
; what makes the count correct.
; CHECK: %[[#]] = OpFunction
; CHECK: %[[#maskedarg:]] = OpBitwiseAnd %[[#i32]] %[[#]] %[[#mask24]]
; CHECK: %[[#countarg:]] = OpExtInst %[[#i32]] %[[#extinst_id]] clz %[[#maskedarg]]
; CHECK: %[[#resultarg:]] = OpISub %[[#i32]] %[[#countarg]] %[[#diff24]]
; CHECK: OpReturnValue %[[#resultarg]]

define spir_func i24 @ctlz_i24_arg(i24 %x) {
  %count = call i24 @llvm.ctlz.i24(i24 %x, i1 false)
  ret i24 %count
}

declare i24 @llvm.ctlz.i24(i24, i1 immarg)

declare i40 @llvm.ctlz.i40(i40, i1 immarg)
