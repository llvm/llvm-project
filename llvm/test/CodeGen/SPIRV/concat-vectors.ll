; RUN: llc -O0 -global-isel -verify-machineinstrs -mtriple=spirv64 %s -o - | FileCheck %s
; spirv-val errors about a 7 element vector.
; TODO: %if spirv-tools %{ llc -O0 -mtriple=spirv64 < %s -o - -filetype=obj | spirv-val %}

; G_CONCAT_VECTORS should select to OpCompositeConstruct, which
; concatenates its vector constituents (each sharing the result component type).

; Note we have to use a non-power of 2 vector length (7 here) that cannot be legalized to actually
; generate the G_CONCAT_VECTOR Opcode. Usually it would be legalized but here it can't because it's
; part of the ABI.

; CHECK: %[[#I8:]] = OpTypeInt 8 0
; CHECK: %[[#V4:]] = OpTypeVector %[[#I8]] 4
; CHECK: %[[#V8:]] = OpTypeVector %[[#I8]] 8
; CHECK: %[[#UNDEF:]] = OpUndef %[[#V4]]
; CHECK: OpFunction
; CHECK: %[[#A:]] = OpFunctionParameter %[[#V4]]
; CHECK: %[[#CONCAT:]] = OpCompositeConstruct %[[#V8]] %[[#A]] %[[#UNDEF]]

define spir_func <7 x i8> @extend_vec4_to_vec7(<4 x i8> %x) {

entry:
  %r = shufflevector <4 x i8> %x, <4 x i8> poison, <7 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison>
  ret <7 x i8> %r
}
