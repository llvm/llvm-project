; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_function_pointers %s -o - | FileCheck %s
; TODO: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_function_pointers %s -o - -filetype=obj | spirv-val %}

declare spir_func i32 @_Z12get_local_idj(i32)

; CHECK: OpName %[[F1:.*]] "func_1"
; CHECK: OpName %[[F2:.*]] "func_2"

; CHECK: %[[I32:.*]] = OpTypeInt 32 0
; CHECK: %[[I64:.*]] = OpTypeInt 64 0

; CHECK: %[[F1]] = OpFunction %[[I32]] None
; CHECK: %[[COMPOSITE_EXTRACT:.*]] = OpCompositeExtract %[[I64]]
; CHECK: OpUConvert %[[I32]] %[[COMPOSITE_EXTRACT]] 

define spir_func i32 @func_1() {
  %id = tail call spir_func i32 @_Z12get_local_idj(i32 0)
  ret i32 %id
}

; CHECK: %[[F2]] = OpFunction %[[I64]] None
; CHECK: OpCompositeExtract %[[I64]] 

define spir_func i64 @func_2() {
  %id = tail call spir_func i64 @_Z12get_local_idj(i32 0)
  ret i64 %id
}
