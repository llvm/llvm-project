; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s --implicit-check-not="OpTypeInt 128" --implicit-check-not="OpTypeInt 512"
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; Bitcasts between bool vectors and other vectors are decomposed per element,
; without an integer as wide as the whole vector. Constant-index extracts only
; compute the extracted lanes.

; CHECK-DAG: %[[#BOOL:]] = OpTypeBool
; CHECK-DAG: %[[#I8:]] = OpTypeInt 8 0
; CHECK-DAG: %[[#I32:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#F32:]] = OpTypeFloat 32
; CHECK-DAG: %[[#V2I8:]] = OpTypeVector %[[#I8]] 2
; CHECK-DAG: %[[#V4I32:]] = OpTypeVector %[[#I32]] 4
; CHECK-DAG: %[[#V16I32:]] = OpTypeVector %[[#I32]] 16
; CHECK-DAG: %[[#V16F32:]] = OpTypeVector %[[#F32]] 16
; CHECK-DAG: %[[#BVEC16:]] = OpTypeVector %[[#BOOL]] 16
; CHECK-DAG: %[[#ZERO8:]] = OpConstantNull %[[#I8]]
; CHECK-DAG: %[[#ONE8:]] = OpConstant %[[#I8]] 1{{$}}
; CHECK-DAG: %[[#TWO8:]] = OpConstant %[[#I8]] 2{{$}}
; CHECK-DAG: %[[#C7_8:]] = OpConstant %[[#I8]] 7{{$}}
; CHECK-DAG: %[[#C128_8:]] = OpConstant %[[#I8]] 128{{$}}
; CHECK-DAG: %[[#ZERO32:]] = OpConstantNull %[[#I32]]
; CHECK-DAG: %[[#ONE32:]] = OpConstant %[[#I32]] 1{{$}}
; CHECK-DAG: %[[#C16_32:]] = OpConstant %[[#I32]] 16{{$}}
; CHECK-DAG: %[[#SIGN32:]] = OpConstant %[[#I32]] 2147483648{{$}}

; bitcast <16 x i32> to <512 x i1>, reading lane 0.
;
; CHECK:     OpFunction %[[#BOOL]]
; CHECK:     %[[#L0_ARG:]] = OpFunctionParameter %[[#V16I32]]
; CHECK:     OpLabel
; CHECK-NEXT:%[[#L0_E:]] = OpCompositeExtract %[[#I32]] %[[#L0_ARG]] 0
; CHECK-NEXT:%[[#L0_A:]] = OpBitwiseAnd %[[#I32]] %[[#L0_E]] %[[#ONE32]]
; CHECK-NEXT:%[[#L0_C:]] = OpINotEqual %[[#BOOL]] %[[#L0_A]] %[[#ZERO32]]
; CHECK-NEXT:OpReturnValue %[[#L0_C]]
; CHECK-NEXT:OpFunctionEnd
define i1 @v16i32_lane0(<16 x i32> %v) {
  %b = bitcast <16 x i32> %v to <512 x i1>
  %e = extractelement <512 x i1> %b, i64 0
  ret i1 %e
}

; bitcast <4 x i32> to <128 x i1>, reading lanes on both sides of element
; boundaries.
;
; CHECK:     OpFunction %[[#BOOL]]
; CHECK:     %[[#LS_ARG:]] = OpFunctionParameter %[[#V4I32]]
; CHECK:     OpLabel
; CHECK-NEXT:%[[#LS_E31:]] = OpCompositeExtract %[[#I32]] %[[#LS_ARG]] 0
; CHECK-NEXT:%[[#LS_A31:]] = OpBitwiseAnd %[[#I32]] %[[#LS_E31]] %[[#SIGN32]]
; CHECK-NEXT:%[[#LS_C31:]] = OpINotEqual %[[#BOOL]] %[[#LS_A31]] %[[#ZERO32]]
; CHECK-NEXT:%[[#LS_E32:]] = OpCompositeExtract %[[#I32]] %[[#LS_ARG]] 1
; CHECK-NEXT:%[[#LS_A32:]] = OpBitwiseAnd %[[#I32]] %[[#LS_E32]] %[[#ONE32]]
; CHECK-NEXT:%[[#LS_C32:]] = OpINotEqual %[[#BOOL]] %[[#LS_A32]] %[[#ZERO32]]
; CHECK-NEXT:%[[#LS_E127:]] = OpCompositeExtract %[[#I32]] %[[#LS_ARG]] 3
; CHECK-NEXT:%[[#LS_A127:]] = OpBitwiseAnd %[[#I32]] %[[#LS_E127]] %[[#SIGN32]]
; CHECK-NEXT:%[[#LS_C127:]] = OpINotEqual %[[#BOOL]] %[[#LS_A127]] %[[#ZERO32]]
; CHECK-NEXT:%[[#LS_X:]] = OpLogicalNotEqual %[[#BOOL]] %[[#LS_C31]] %[[#LS_C32]]
; CHECK-NEXT:%[[#LS_Y:]] = OpLogicalNotEqual %[[#BOOL]] %[[#LS_X]] %[[#LS_C127]]
; CHECK-NEXT:OpReturnValue %[[#LS_Y]]
; CHECK-NEXT:OpFunctionEnd
define i1 @v4i32_lanes(<4 x i32> %v) {
  %b = bitcast <4 x i32> %v to <128 x i1>
  %e31 = extractelement <128 x i1> %b, i64 31
  %e32 = extractelement <128 x i1> %b, i64 32
  %e127 = extractelement <128 x i1> %b, i64 127
  %x = xor i1 %e31, %e32
  %y = xor i1 %x, %e127
  ret i1 %y
}

; bitcast <16 x float> to <512 x i1>, reading lane 100: bit 4 of element 3.
;
; CHECK:     OpFunction %[[#BOOL]]
; CHECK:     %[[#FL_ARG:]] = OpFunctionParameter %[[#V16F32]]
; CHECK:     OpLabel
; CHECK-NEXT:%[[#FL_I:]] = OpBitcast %[[#V16I32]] %[[#FL_ARG]]
; CHECK-NEXT:%[[#FL_E:]] = OpCompositeExtract %[[#I32]] %[[#FL_I]] 3
; CHECK-NEXT:%[[#FL_A:]] = OpBitwiseAnd %[[#I32]] %[[#FL_E]] %[[#C16_32]]
; CHECK-NEXT:%[[#FL_C:]] = OpINotEqual %[[#BOOL]] %[[#FL_A]] %[[#ZERO32]]
; CHECK-NEXT:OpReturnValue %[[#FL_C]]
; CHECK-NEXT:OpFunctionEnd
define i1 @v16f32_lane100(<16 x float> %v) {
  %b = bitcast <16 x float> %v to <512 x i1>
  %e = extractelement <512 x i1> %b, i64 100
  ret i1 %e
}

; bitcast <2 x i8> to <16 x i1> with a variable-index extract, which needs all
; lanes, and a constant-index extract of lane 9, which is still computed
; directly.
;
; CHECK:     OpFunction %[[#BOOL]]
; CHECK:     %[[#DYN_ARG:]] = OpFunctionParameter %[[#V2I8]]
; CHECK:     %[[#DYN_IDX:]] = OpFunctionParameter %[[#I32]]
; CHECK:     OpLabel
; CHECK:     %[[#DYN_VEC:]] = OpCompositeInsert %[[#BVEC16]] %{{.*}} %{{.*}} 15
; CHECK-NEXT:%[[#DYN_E:]] = OpVectorExtractDynamic %[[#BOOL]] %[[#DYN_VEC]] %[[#DYN_IDX]]
; CHECK-NEXT:%[[#DYN_E9:]] = OpCompositeExtract %[[#I8]] %[[#DYN_ARG]] 1
; CHECK-NEXT:%[[#DYN_A9:]] = OpBitwiseAnd %[[#I8]] %[[#DYN_E9]] %[[#TWO8]]
; CHECK-NEXT:%[[#DYN_C9:]] = OpINotEqual %[[#BOOL]] %[[#DYN_A9]] %[[#ZERO8]]
; CHECK-NEXT:%[[#DYN_X:]] = OpLogicalNotEqual %[[#BOOL]] %[[#DYN_E]] %[[#DYN_C9]]
; CHECK-NEXT:OpReturnValue %[[#DYN_X]]
; CHECK-NEXT:OpFunctionEnd
define i1 @v2i8_dynamic_lane(<2 x i8> %v, i32 %i) {
  %b = bitcast <2 x i8> %v to <16 x i1>
  %e = extractelement <16 x i1> %b, i32 %i
  %e9 = extractelement <16 x i1> %b, i32 9
  %x = xor i1 %e, %e9
  ret i1 %x
}

; bitcast <4 x float> to <128 x i1> and back to <4 x i32>, with all lanes used.
;
; CHECK:     OpFunction %[[#V4I32]]
; CHECK:     %[[#RT_ARG:]] = OpFunctionParameter
; CHECK:     OpLabel
; CHECK:     %[[#RT_I:]] = OpBitcast %[[#V4I32]] %[[#RT_ARG]]
; CHECK:     OpCompositeExtract %[[#I32]] %[[#RT_I]] 0
; CHECK:     OpCompositeExtract %[[#I32]] %[[#RT_I]] 3
; CHECK:     OpBitwiseAnd %[[#I32]] %{{.*}} %[[#SIGN32]]
; CHECK:     OpLogicalNot
; CHECK:     OpCompositeInsert %[[#V4I32]] %{{.*}} %{{.*}} 3
; CHECK:     OpReturnValue
; CHECK:     OpFunctionEnd
define <4 x i32> @v4f32_roundtrip(<4 x float> %v) {
  %b = bitcast <4 x float> %v to <128 x i1>
  %n = xor <128 x i1> %b, splat (i1 true)
  %r = bitcast <128 x i1> %n to <4 x i32>
  ret <4 x i32> %r
}

; bitcast <16 x i32> to <512 x i1> and back, with all lanes used.
;
; CHECK:     OpFunction %[[#V16I32]]
; CHECK:     %[[#RT512_ARG:]] = OpFunctionParameter %[[#V16I32]]
; CHECK:     OpLabel
; CHECK:     OpCompositeExtract %[[#I32]] %[[#RT512_ARG]] 15
; CHECK:     OpCompositeInsert %[[#V16I32]] %{{.*}} %{{.*}} 15
; CHECK:     OpReturnValue
; CHECK:     OpFunctionEnd
define <16 x i32> @v16i32_roundtrip(<16 x i32> %v) {
  %b = bitcast <16 x i32> %v to <512 x i1>
  %n = xor <512 x i1> %b, splat (i1 true)
  %r = bitcast <512 x i1> %n to <16 x i32>
  ret <16 x i32> %r
}

; bitcast <2 x i8> to <16 x i1>: lanes 0 to 7 are bits of element 0, lanes 8
; to 15 bits of element 1.
;
; CHECK:     OpFunction %[[#BVEC16]]
; CHECK:     %[[#V2B_ARG:]] = OpFunctionParameter %[[#V2I8]]
; CHECK:     OpLabel
; CHECK-NEXT:%[[#V2B_E0:]] = OpCompositeExtract %[[#I8]] %[[#V2B_ARG]] 0
; CHECK-NEXT:%[[#V2B_A0:]] = OpBitwiseAnd %[[#I8]] %[[#V2B_E0]] %[[#ONE8]]
; CHECK-NEXT:%[[#V2B_C0:]] = OpINotEqual %[[#BOOL]] %[[#V2B_A0]] %[[#ZERO8]]
; CHECK-NEXT:%[[#V2B_I0:]] = OpCompositeInsert %[[#BVEC16]] %[[#V2B_C0]] %{{.*}} 0
; CHECK:     OpCompositeInsert %[[#BVEC16]] %{{.*}} %{{.*}} 6
; CHECK-NEXT:%[[#V2B_E7:]] = OpCompositeExtract %[[#I8]] %[[#V2B_ARG]] 0
; CHECK-NEXT:%[[#V2B_A7:]] = OpBitwiseAnd %[[#I8]] %[[#V2B_E7]] %[[#C128_8]]
; CHECK-NEXT:%[[#V2B_C7:]] = OpINotEqual %[[#BOOL]] %[[#V2B_A7]] %[[#ZERO8]]
; CHECK-NEXT:%[[#V2B_I7:]] = OpCompositeInsert %[[#BVEC16]] %[[#V2B_C7]] %{{.*}} 7
; CHECK-NEXT:%[[#V2B_E8:]] = OpCompositeExtract %[[#I8]] %[[#V2B_ARG]] 1
; CHECK-NEXT:%[[#V2B_A8:]] = OpBitwiseAnd %[[#I8]] %[[#V2B_E8]] %[[#ONE8]]
; CHECK-NEXT:%[[#V2B_C8:]] = OpINotEqual %[[#BOOL]] %[[#V2B_A8]] %[[#ZERO8]]
; CHECK-NEXT:%[[#V2B_I8:]] = OpCompositeInsert %[[#BVEC16]] %[[#V2B_C8]] %[[#V2B_I7]] 8
; CHECK:     %[[#V2B_I14:]] = OpCompositeInsert %[[#BVEC16]] %{{.*}} %{{.*}} 14
; CHECK-NEXT:%[[#V2B_E15:]] = OpCompositeExtract %[[#I8]] %[[#V2B_ARG]] 1
; CHECK-NEXT:%[[#V2B_A15:]] = OpBitwiseAnd %[[#I8]] %[[#V2B_E15]] %[[#C128_8]]
; CHECK-NEXT:%[[#V2B_C15:]] = OpINotEqual %[[#BOOL]] %[[#V2B_A15]] %[[#ZERO8]]
; CHECK-NEXT:%[[#V2B_I15:]] = OpCompositeInsert %[[#BVEC16]] %[[#V2B_C15]] %[[#V2B_I14]] 15
; CHECK-NEXT:OpReturnValue %[[#V2B_I15]]
; CHECK-NEXT:OpFunctionEnd
define <16 x i1> @v2i8_to_boolvec(<2 x i8> %v) {
  %b = bitcast <2 x i8> %v to <16 x i1>
  ret <16 x i1> %b
}

; bitcast <16 x i1> to <2 x i8>: element 0 is built from lanes 0 to 7, element 1
; from lanes 8 to 15.
;
; CHECK:     OpFunction %[[#V2I8]]
; CHECK:     %[[#B2V_ARG:]] = OpFunctionParameter %[[#BVEC16]]
; CHECK:     OpLabel
; CHECK:     OpCompositeExtract %[[#BOOL]] %[[#B2V_ARG]] 6
; CHECK-NEXT:OpSelect
; CHECK-NEXT:OpShiftLeftLogical
; CHECK-NEXT:%[[#B2V_O6:]] = OpBitwiseOr %[[#I8]]
; CHECK-NEXT:%[[#B2V_L7:]] = OpCompositeExtract %[[#BOOL]] %[[#B2V_ARG]] 7
; CHECK-NEXT:%[[#B2V_S7:]] = OpSelect %[[#I8]] %[[#B2V_L7]] %[[#ONE8]] %[[#ZERO8]]
; CHECK-NEXT:%[[#B2V_H7:]] = OpShiftLeftLogical %[[#I8]] %[[#B2V_S7]] %[[#C7_8]]
; CHECK-NEXT:%[[#B2V_O7:]] = OpBitwiseOr %[[#I8]] %[[#B2V_O6]] %[[#B2V_H7]]
; CHECK-NEXT:%[[#B2V_V0:]] = OpCompositeInsert %[[#V2I8]] %[[#B2V_O7]] %{{.*}} 0
; CHECK-NEXT:%[[#B2V_L8:]] = OpCompositeExtract %[[#BOOL]] %[[#B2V_ARG]] 8
; CHECK-NEXT:%[[#B2V_S8:]] = OpSelect %[[#I8]] %[[#B2V_L8]] %[[#ONE8]] %[[#ZERO8]]
; CHECK-NEXT:%[[#B2V_O8:]] = OpBitwiseOr %[[#I8]] %[[#ZERO8]] %[[#B2V_S8]]
; CHECK-NEXT:%[[#B2V_L9:]] = OpCompositeExtract %[[#BOOL]] %[[#B2V_ARG]] 9
; CHECK-NEXT:%[[#B2V_S9:]] = OpSelect %[[#I8]] %[[#B2V_L9]] %[[#ONE8]] %[[#ZERO8]]
; CHECK-NEXT:%[[#B2V_H9:]] = OpShiftLeftLogical %[[#I8]] %[[#B2V_S9]] %[[#ONE8]]
; CHECK-NEXT:%[[#B2V_O9:]] = OpBitwiseOr %[[#I8]] %[[#B2V_O8]] %[[#B2V_H9]]
; CHECK:     OpCompositeExtract %[[#BOOL]] %[[#B2V_ARG]] 14
; CHECK-NEXT:OpSelect
; CHECK-NEXT:OpShiftLeftLogical
; CHECK-NEXT:%[[#B2V_O14:]] = OpBitwiseOr %[[#I8]]
; CHECK-NEXT:%[[#B2V_L15:]] = OpCompositeExtract %[[#BOOL]] %[[#B2V_ARG]] 15
; CHECK-NEXT:%[[#B2V_S15:]] = OpSelect %[[#I8]] %[[#B2V_L15]] %[[#ONE8]] %[[#ZERO8]]
; CHECK-NEXT:%[[#B2V_H15:]] = OpShiftLeftLogical %[[#I8]] %[[#B2V_S15]] %[[#C7_8]]
; CHECK-NEXT:%[[#B2V_O15:]] = OpBitwiseOr %[[#I8]] %[[#B2V_O14]] %[[#B2V_H15]]
; CHECK-NEXT:%[[#B2V_V1:]] = OpCompositeInsert %[[#V2I8]] %[[#B2V_O15]] %[[#B2V_V0]] 1
; CHECK-NEXT:OpReturnValue %[[#B2V_V1]]
; CHECK-NEXT:OpFunctionEnd
define <2 x i8> @boolvec_to_v2i8(<16 x i1> %v) {
  %b = bitcast <16 x i1> %v to <2 x i8>
  ret <2 x i8> %b
}
