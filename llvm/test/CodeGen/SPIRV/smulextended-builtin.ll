; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

%i8struct = type {i8, i8}
%i16struct = type {i16, i16}
%i32struct = type {i32, i32}
%i64struct = type {i64, i64}
%vecstruct = type {<4 x i32>, <4 x i32>}

; CHECK-SPIRV-DAG:                     %[[#uchar:]] = OpTypeInt 8
; CHECK-SPIRV-DAG:                    %[[#ushort:]] = OpTypeInt 16
; CHECK-SPIRV-DAG:                      %[[#uint:]] = OpTypeInt 32
; CHECK-SPIRV-DAG:                     %[[#ulong:]] = OpTypeInt 64
; CHECK-SPIRV-DAG:                  %[[#i8struct:]] = OpTypeStruct %[[#uchar]] %[[#uchar]]
; CHECK-SPIRV-DAG:                 %[[#i16struct:]] = OpTypeStruct %[[#ushort]] %[[#ushort]]
; CHECK-SPIRV-DAG:                 %[[#i32struct:]] = OpTypeStruct %[[#uint]] %[[#uint]]
; CHECK-SPIRV-DAG:                 %[[#i64struct:]] = OpTypeStruct %[[#ulong]] %[[#ulong]]
; CHECK-SPIRV-DAG:                    %[[#v4uint:]] = OpTypeVector %[[#uint]] 4
; CHECK-SPIRV-DAG:                 %[[#vecstruct:]] = OpTypeStruct %[[#v4uint]] %[[#v4uint]]

define spir_func void @test_builtin_smulext_sret(i32 %a, i32 %b) {
  entry:
  %0 = alloca %i32struct
  call void @__spirv_SMulExtended(ptr sret (%i32struct) %0, i32 %a, i32 %b)
  ret void
}
; CHECK-SPIRV:                %[[#]] = OpFunction %[[#]] None %[[#]] ; -- Begin function test_builtin_smulext_sret
; CHECK-SPIRV-NEXT:        %[[#a_s:]] = OpFunctionParameter %[[#uint]]
; CHECK-SPIRV-NEXT:        %[[#b_s:]] = OpFunctionParameter %[[#uint]]
; CHECK-SPIRV-NEXT:    %[[#entry_s:]] = OpLabel
; CHECK-SPIRV:           %[[#var_s:]] = OpVariable %[[#_ptr_Function_i32struct:]] Function
; CHECK-SPIRV:           %[[#res_s:]] = OpSMulExtended %[[#i32struct]] %[[#a_s]] %[[#b_s]]
; CHECK-SPIRV-NEXT:                     OpStore %[[#var_s]] %[[#res_s]]
; CHECK-SPIRV-NEXT:                     OpReturn
; CHECK-SPIRV-NEXT:                     OpFunctionEnd

define spir_func %i8struct @test_builtin_smulextcc(i8 %a, i8 %b) {
  entry:
  %0 = call %i8struct @_Z20__spirv_SMulExtendedcc(i8 %a, i8 %b)
  ret %i8struct %0
}
; CHECK-SPIRV:             %[[#a:]] = OpFunctionParameter %[[#uchar]]
; CHECK-SPIRV-NEXT:        %[[#b:]] = OpFunctionParameter %[[#uchar]]
; CHECK-SPIRV-NEXT:    %[[#entry:]] = OpLabel
; CHECK-SPIRV-NEXT:      %[[#res:]] = OpSMulExtended %[[#i8struct]] %[[#a]] %[[#b]]
; CHECK-SPIRV-NEXT:                   OpReturnValue %[[#res]]
; CHECK-SPIRV-NEXT:                   OpFunctionEnd

define spir_func %i16struct @test_builtin_smulextss(i16 %a, i16 %b) {
  entry:
  %0 = call %i16struct @_Z20__spirv_SMulExtendedss(i16 %a, i16 %b)
  ret %i16struct %0
}
; CHECK-SPIRV:             %[[#a_0:]] = OpFunctionParameter %[[#ushort]]
; CHECK-SPIRV-NEXT:        %[[#b_0:]] = OpFunctionParameter %[[#ushort]]
; CHECK-SPIRV-NEXT:    %[[#entry_0:]] = OpLabel
; CHECK-SPIRV-NEXT:      %[[#res_0:]] = OpSMulExtended %[[#i16struct]] %[[#a_0]] %[[#b_0]]
; CHECK-SPIRV-NEXT:                     OpReturnValue %[[#res_0]]
; CHECK-SPIRV-NEXT:                     OpFunctionEnd

define spir_func %i32struct @test_builtin_smulextii(i32 %a, i32 %b) {
  entry:
  %0 = call %i32struct @_Z20__spirv_SMulExtendedii(i32 %a, i32 %b)
  ret %i32struct %0
}
; CHECK-SPIRV:             %[[#a_1:]] = OpFunctionParameter %[[#uint]]
; CHECK-SPIRV-NEXT:        %[[#b_1:]] = OpFunctionParameter %[[#uint]]
; CHECK-SPIRV-NEXT:    %[[#entry_1:]] = OpLabel
; CHECK-SPIRV-NEXT:      %[[#res_1:]] = OpSMulExtended %[[#i32struct]] %[[#a_1]] %[[#b_1]]
; CHECK-SPIRV-NEXT:                     OpReturnValue %[[#res_1]]
; CHECK-SPIRV-NEXT:                     OpFunctionEnd

define spir_func %i64struct @test_builtin_smulextll(i64 %a, i64 %b) {
  entry:
  %0 = call %i64struct @_Z20__spirv_SMulExtendedll(i64 %a, i64 %b)
  ret %i64struct %0
}
; CHECK-SPIRV:             %[[#a_2:]] = OpFunctionParameter %[[#ulong]]
; CHECK-SPIRV-NEXT:        %[[#b_2:]] = OpFunctionParameter %[[#ulong]]
; CHECK-SPIRV-NEXT:    %[[#entry_2:]] = OpLabel
; CHECK-SPIRV-NEXT:      %[[#res_2:]] = OpSMulExtended %[[#i64struct]] %[[#a_2]] %[[#b_2]]
; CHECK-SPIRV-NEXT:                     OpReturnValue %[[#res_2]]
; CHECK-SPIRV-NEXT:                     OpFunctionEnd

define spir_func %vecstruct @test_builtin_smulextDv4_xS_(<4 x i32> %a, <4 x i32> %b) {
  entry:
  %0 = call %vecstruct @_Z20__spirv_SMulExtendedDv4_iS_(<4 x i32> %a, <4 x i32> %b)
  ret %vecstruct %0
}
; CHECK-SPIRV:             %[[#a_3:]] = OpFunctionParameter %[[#v4uint]]
; CHECK-SPIRV-NEXT:        %[[#b_3:]] = OpFunctionParameter %[[#v4uint]]
; CHECK-SPIRV-NEXT:    %[[#entry_3:]] = OpLabel
; CHECK-SPIRV-NEXT:      %[[#res_3:]] = OpSMulExtended %[[#vecstruct]] %[[#a_3]] %[[#b_3]]
; CHECK-SPIRV-NEXT:                     OpReturnValue %[[#res_3]]
; CHECK-SPIRV-NEXT:                     OpFunctionEnd

define spir_func %i32struct @test_builtin_smulext_same_arg(i32 %a) {
  entry:
  %0 = call %i32struct @_Z20__spirv_SMulExtendedii(i32 %a, i32 %a)
  ret %i32struct %0
}
; CHECK-SPIRV:             %[[#a_4:]] = OpFunctionParameter %[[#uint]]
; CHECK-SPIRV-NEXT:    %[[#entry_4:]] = OpLabel
; CHECK-SPIRV-NEXT:      %[[#res_4:]] = OpSMulExtended %[[#i32struct]] %[[#a_4]] %[[#a_4]]
; CHECK-SPIRV-NEXT:                     OpReturnValue %[[#res_4]]
; CHECK-SPIRV-NEXT:                     OpFunctionEnd

declare %i8struct @_Z20__spirv_SMulExtendedcc(i8, i8)
declare %i16struct @_Z20__spirv_SMulExtendedss(i16, i16)
declare %i32struct @_Z20__spirv_SMulExtendedii(i32, i32)
declare %i64struct @_Z20__spirv_SMulExtendedll(i64, i64)
declare %vecstruct @_Z20__spirv_SMulExtendedDv4_iS_(<4 x i32>, <4 x i32>)
declare void @__spirv_SMulExtended(ptr sret (%i32struct), i32, i32)
