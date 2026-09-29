; Check that convert_ builtins pick the signed/unsigned opcode from the source
; operand signedness.

; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: OpName %[[#StoS:]] "s_to_s"
; CHECK-DAG: OpName %[[#StoU:]] "s_to_u"
; CHECK-DAG: OpName %[[#UtoS:]] "u_to_s"
; CHECK-DAG: OpName %[[#UtoU:]] "u_to_u"
; CHECK-DAG: OpName %[[#StoF:]] "s_to_f"
; CHECK-DAG: OpName %[[#UtoF:]] "u_to_f"

; signed source -> signed dest: sign-extend.
; CHECK: %[[#StoS]] = OpFunction
; CHECK: OpSConvert
define spir_func void @s_to_s(i32 noundef %x) {
  call spir_func i64 @_Z12convert_longi(i32 noundef %x)
  ret void
}

; signed source -> unsigned dest: sign-extend.
; CHECK: %[[#StoU]] = OpFunction
; CHECK: OpSConvert
define spir_func void @s_to_u(i32 noundef %x) {
  call spir_func i64 @_Z13convert_ulongi(i32 noundef %x)
  ret void
}

; unsigned source -> signed dest: zero-extend.
; CHECK: %[[#UtoS]] = OpFunction
; CHECK: OpUConvert
define spir_func void @u_to_s(i32 noundef %x) {
  call spir_func i64 @_Z12convert_longj(i32 noundef %x)
  ret void
}

; unsigned source -> unsigned dest: zero-extend.
; CHECK: %[[#UtoU]] = OpFunction
; CHECK: OpUConvert
define spir_func void @u_to_u(i32 noundef %x) {
  call spir_func i64 @_Z13convert_ulongj(i32 noundef %x)
  ret void
}

; signed source -> float.
; CHECK: %[[#StoF]] = OpFunction
; CHECK: OpConvertSToF
define spir_func void @s_to_f(i32 noundef %x) {
  call spir_func float @_Z13convert_floati(i32 noundef %x)
  ret void
}

; unsigned source -> float.
; CHECK: %[[#UtoF]] = OpFunction
; CHECK: OpConvertUToF
define spir_func void @u_to_f(i32 noundef %x) {
  call spir_func float @_Z13convert_floatj(i32 noundef %x)
  ret void
}

declare spir_func i64 @_Z12convert_longi(i32 noundef)
declare spir_func i64 @_Z13convert_ulongi(i32 noundef)
declare spir_func i64 @_Z12convert_longj(i32 noundef)
declare spir_func i64 @_Z13convert_ulongj(i32 noundef)
declare spir_func float @_Z13convert_floati(i32 noundef)
declare spir_func float @_Z13convert_floatj(i32 noundef)
