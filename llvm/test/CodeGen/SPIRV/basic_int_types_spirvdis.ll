; REQUIRES: spirv-tools
; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - --filetype=obj | spirv-dis | FileCheck %s
; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - --filetype=obj | spirv-dis | FileCheck %s

define void @main() {
entry:
; CHECK: %int16_t_Val = OpVariable %_ptr_Function_ushort Function
  %int16_t_Val = alloca i16, align 2

; CHECK: %int_Val = OpVariable %_ptr_Function_uint Function
  %int_Val = alloca i32, align 4

; CHECK: %int64_t_Val = OpVariable %_ptr_Function_ulong Function
  %int64_t_Val = alloca i64, align 8

; CHECK: %int16_t2_Val = OpVariable %_ptr_Function_v2ushort Function
  %int16_t2_Val = alloca <2 x i16>, align 4

; CHECK: %int16_t3_Val = OpVariable %_ptr_Function_v3ushort Function
  %int16_t3_Val = alloca <3 x i16>, align 8

; CHECK: %int16_t4_Val = OpVariable %_ptr_Function_v4ushort Function
  %int16_t4_Val = alloca <4 x i16>, align 8

; CHECK: %int2_Val = OpVariable %_ptr_Function_v2uint Function
  %int2_Val = alloca <2 x i32>, align 8

; CHECK: %int3_Val = OpVariable %_ptr_Function_v3uint Function
  %int3_Val = alloca <3 x i32>, align 16

; CHECK: %int4_Val = OpVariable %_ptr_Function_v4uint Function
  %int4_Val = alloca <4 x i32>, align 16

; CHECK: %int64_t2_Val = OpVariable %_ptr_Function_v2ulong Function
  %int64_t2_Val = alloca <2 x i64>, align 16

; CHECK: %int64_t3_Val = OpVariable %_ptr_Function_v3ulong Function
  %int64_t3_Val = alloca <3 x i64>, align 32

; CHECK: %int64_t4_Val = OpVariable %_ptr_Function_v4ulong Function
  %int64_t4_Val = alloca <4 x i64>, align 32

  ret void
}
