; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_EXT_long_vector,+SPV_ALTERA_arbitrary_precision_integers %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_EXT_long_vector,+SPV_ALTERA_arbitrary_precision_integers %s -o - -filetype=obj | spirv-val %}

; Verify that bitcasts between bool vectors and other types are decomposed
; into element-wise operations instead of generating OpBitcast, which is
; invalid for OpTypeBool.

; CHECK-DAG: %[[#I32:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#ONE:]] = OpConstant %[[#I32]] 1
; CHECK-DAG: %[[#SEVENTEEN:]] = OpConstant %[[#I32]] 17
; CHECK-DAG: %[[#BOOL:]] = OpTypeBool
; CHECK-DAG: %[[#BVEC8:]] = OpTypeVector %[[#BOOL]] 8
; CHECK-DAG: %[[#BVEC17:]] = OpTypeVectorIdEXT %[[#BOOL]] %[[#SEVENTEEN]]
; CHECK-DAG: %[[#I8:]] = OpTypeInt 8 0
; CHECK-DAG: %[[#I17:]] = OpTypeInt 17 0
; CHECK-DAG: %[[#VEC1_I8:]] = OpTypeVectorIdEXT %[[#I8]] %[[#ONE]]

; CHECK-DAG: %[[#ZERO:]] = OpConstantNull %[[#I8]]
; CHECK-DAG: %[[#ONE:]] = OpConstant %[[#I8]] 1{{$}}
; CHECK-DAG: %[[#TWO:]] = OpConstant %[[#I8]] 2{{$}}
; CHECK-DAG: %[[#FOUR:]] = OpConstant %[[#I8]] 4{{$}}
; CHECK-DAG: %[[#EIGHT:]] = OpConstant %[[#I8]] 8{{$}}
; CHECK-DAG: %[[#C16:]] = OpConstant %[[#I8]] 16{{$}}
; CHECK-DAG: %[[#C32:]] = OpConstant %[[#I8]] 32{{$}}
; CHECK-DAG: %[[#C64:]] = OpConstant %[[#I8]] 64{{$}}
; CHECK-DAG: %[[#C128:]] = OpConstant %[[#I8]] 128{{$}}
; CHECK-DAG: %[[#C3:]] = OpConstant %[[#I8]] 3{{$}}
; CHECK-DAG: %[[#C5:]] = OpConstant %[[#I8]] 5{{$}}
; CHECK-DAG: %[[#C6:]] = OpConstant %[[#I8]] 6{{$}}
; CHECK-DAG: %[[#C7:]] = OpConstant %[[#I8]] 7{{$}}

; CHECK-DAG: %[[#ZERO17:]] = OpConstantNull %[[#I17]]
; CHECK-DAG: %[[#ONE17:]] = OpConstant %[[#I17]] 1{{$}}
; CHECK-DAG: %[[#TWO17:]] = OpConstant %[[#I17]] 2{{$}}
; CHECK-DAG: %[[#C3_17:]] = OpConstant %[[#I17]] 3{{$}}
; CHECK-DAG: %[[#C4_17:]] = OpConstant %[[#I17]] 4{{$}}
; CHECK-DAG: %[[#C5_17:]] = OpConstant %[[#I17]] 5{{$}}
; CHECK-DAG: %[[#C6_17:]] = OpConstant %[[#I17]] 6{{$}}
; CHECK-DAG: %[[#C7_17:]] = OpConstant %[[#I17]] 7{{$}}
; CHECK-DAG: %[[#C8_17:]] = OpConstant %[[#I17]] 8{{$}}
; CHECK-DAG: %[[#C9_17:]] = OpConstant %[[#I17]] 9{{$}}
; CHECK-DAG: %[[#C10_17:]] = OpConstant %[[#I17]] 10{{$}}
; CHECK-DAG: %[[#C11_17:]] = OpConstant %[[#I17]] 11{{$}}
; CHECK-DAG: %[[#C12_17:]] = OpConstant %[[#I17]] 12{{$}}
; CHECK-DAG: %[[#C13_17:]] = OpConstant %[[#I17]] 13{{$}}
; CHECK-DAG: %[[#C14_17:]] = OpConstant %[[#I17]] 14{{$}}
; CHECK-DAG: %[[#C15_17:]] = OpConstant %[[#I17]] 15{{$}}
; CHECK-DAG: %[[#C16_17:]] = OpConstant %[[#I17]] 16{{$}}
; CHECK-DAG: %[[#C32_17:]] = OpConstant %[[#I17]] 32{{$}}
; CHECK-DAG: %[[#C64_17:]] = OpConstant %[[#I17]] 64{{$}}
; CHECK-DAG: %[[#C128_17:]] = OpConstant %[[#I17]] 128{{$}}
; CHECK-DAG: %[[#C256_17:]] = OpConstant %[[#I17]] 256{{$}}
; CHECK-DAG: %[[#C512_17:]] = OpConstant %[[#I17]] 512{{$}}
; CHECK-DAG: %[[#C1024_17:]] = OpConstant %[[#I17]] 1024{{$}}
; CHECK-DAG: %[[#C2048_17:]] = OpConstant %[[#I17]] 2048{{$}}
; CHECK-DAG: %[[#C4096_17:]] = OpConstant %[[#I17]] 4096{{$}}
; CHECK-DAG: %[[#C8192_17:]] = OpConstant %[[#I17]] 8192{{$}}
; CHECK-DAG: %[[#C16384_17:]] = OpConstant %[[#I17]] 16384{{$}}
; CHECK-DAG: %[[#C32768_17:]] = OpConstant %[[#I17]] 32768{{$}}
; CHECK-DAG: %[[#C65536_17:]] = OpConstant %[[#I17]] 65536{{$}}


; bitcast <8 x i1> to i8
; Extracts each bool, zero-extends via OpSelect, shifts into position, and ORs.
;
; CHECK:   %[[#B2S:]] = OpFunction %[[#I8]]
; CHECK:   %[[#B2S_ARG:]] = OpFunctionParameter %[[#BVEC8]]
; CHECK:   OpLabel
;
; CHECK:   %[[#E0:]] = OpCompositeExtract %[[#BOOL]] %[[#B2S_ARG]] 0
; CHECK:   %[[#S0:]] = OpSelect %[[#I8]] %[[#E0]] %[[#ONE]] %[[#ZERO]]
; CHECK:   %[[#OR0:]] = OpBitwiseOr %[[#I8]] %[[#ZERO]] %[[#S0]]
;
; CHECK:   %[[#E1:]] = OpCompositeExtract %[[#BOOL]] %[[#B2S_ARG]] 1
; CHECK:   %[[#S1:]] = OpSelect %[[#I8]] %[[#E1]] %[[#ONE]] %[[#ZERO]]
; CHECK:   %[[#SHL1:]] = OpShiftLeftLogical %[[#I8]] %[[#S1]] %[[#ONE]]
; CHECK:   %[[#OR1:]] = OpBitwiseOr %[[#I8]] %[[#OR0]] %[[#SHL1]]
;
; CHECK:   %[[#E2:]] = OpCompositeExtract %[[#BOOL]] %[[#B2S_ARG]] 2
; CHECK:   %[[#S2:]] = OpSelect %[[#I8]] %[[#E2]] %[[#ONE]] %[[#ZERO]]
; CHECK:   %[[#SHL2:]] = OpShiftLeftLogical %[[#I8]] %[[#S2]] %[[#TWO]]
; CHECK:   %[[#OR2:]] = OpBitwiseOr %[[#I8]] %[[#OR1]] %[[#SHL2]]
;
; CHECK:   OpCompositeExtract %[[#BOOL]] %[[#B2S_ARG]] 3
; CHECK:   OpShiftLeftLogical %[[#I8]] %{{.*}} %[[#C3]]
; CHECK:   %[[#OR3:]] = OpBitwiseOr %[[#I8]]
; CHECK:   OpCompositeExtract %[[#BOOL]] %[[#B2S_ARG]] 4
; CHECK:   OpShiftLeftLogical %[[#I8]] %{{.*}} %[[#FOUR]]
; CHECK:   %[[#OR4:]] = OpBitwiseOr %[[#I8]]
; CHECK:   OpCompositeExtract %[[#BOOL]] %[[#B2S_ARG]] 5
; CHECK:   OpShiftLeftLogical %[[#I8]] %{{.*}} %[[#C5]]
; CHECK:   %[[#OR5:]] = OpBitwiseOr %[[#I8]]
; CHECK:   OpCompositeExtract %[[#BOOL]] %[[#B2S_ARG]] 6
; CHECK:   OpShiftLeftLogical %[[#I8]] %{{.*}} %[[#C6]]
; CHECK:   %[[#OR6:]] = OpBitwiseOr %[[#I8]]
;
; CHECK:   %[[#E7:]] = OpCompositeExtract %[[#BOOL]] %[[#B2S_ARG]] 7
; CHECK:   %[[#S7:]] = OpSelect %[[#I8]] %[[#E7]] %[[#ONE]] %[[#ZERO]]
; CHECK:   %[[#SHL7:]] = OpShiftLeftLogical %[[#I8]] %[[#S7]] %[[#C7]]
; CHECK:   %[[#OR7:]] = OpBitwiseOr %[[#I8]] %[[#OR6]] %[[#SHL7]]
;
; CHECK:   OpReturnValue %[[#OR7]]
define i8 @boolvec_to_scalar(<8 x i1> %v) {
  %r = bitcast <8 x i1> %v to i8
  ret i8 %r
}

; bitcast <8 x i1> to <1 x i8>
;
; CHECK:   %[[#B2V:]] = OpFunction %[[#VEC1_I8]]
; CHECK:   %[[#B2V_ARG:]] = OpFunctionParameter %[[#BVEC8]]
; CHECK:   OpLabel
;
; CHECK:   %[[#B2V_E0:]] = OpCompositeExtract %[[#BOOL]] %[[#B2V_ARG]] 0
; CHECK:   %[[#B2V_S0:]] = OpSelect %[[#I8]] %[[#B2V_E0]] %[[#ONE]] %[[#ZERO]]
; CHECK:   %[[#B2V_OR0:]] = OpBitwiseOr %[[#I8]] %[[#ZERO]] %[[#B2V_S0]]
;
; CHECK:   %[[#B2V_E1:]] = OpCompositeExtract %[[#BOOL]] %[[#B2V_ARG]] 1
; CHECK:   %[[#B2V_S1:]] = OpSelect %[[#I8]] %[[#B2V_E1]] %[[#ONE]] %[[#ZERO]]
; CHECK:   %[[#B2V_SHL1:]] = OpShiftLeftLogical %[[#I8]] %[[#B2V_S1]] %[[#ONE]]
; CHECK:   %[[#B2V_OR1:]] = OpBitwiseOr %[[#I8]] %[[#B2V_OR0]] %[[#B2V_SHL1]]
;
; CHECK:   %[[#B2V_E2:]] = OpCompositeExtract %[[#BOOL]] %[[#B2V_ARG]] 2
; CHECK:   %[[#B2V_S2:]] = OpSelect %[[#I8]] %[[#B2V_E2]] %[[#ONE]] %[[#ZERO]]
; CHECK:   %[[#B2V_SHL2:]] = OpShiftLeftLogical %[[#I8]] %[[#B2V_S2]] %[[#TWO]]
; CHECK:   %[[#B2V_OR2:]] = OpBitwiseOr %[[#I8]] %[[#B2V_OR1]] %[[#B2V_SHL2]]
;
; CHECK:   OpCompositeExtract %[[#BOOL]] %[[#B2V_ARG]] 3
; CHECK:   OpShiftLeftLogical %[[#I8]] %{{.*}} %[[#C3]]
; CHECK:   %[[#B2V_OR3:]] = OpBitwiseOr %[[#I8]]
; CHECK:   OpCompositeExtract %[[#BOOL]] %[[#B2V_ARG]] 4
; CHECK:   OpShiftLeftLogical %[[#I8]] %{{.*}} %[[#FOUR]]
; CHECK:   %[[#B2V_OR4:]] = OpBitwiseOr %[[#I8]]
; CHECK:   OpCompositeExtract %[[#BOOL]] %[[#B2V_ARG]] 5
; CHECK:   OpShiftLeftLogical %[[#I8]] %{{.*}} %[[#C5]]
; CHECK:   %[[#B2V_OR5:]] = OpBitwiseOr %[[#I8]]
; CHECK:   OpCompositeExtract %[[#BOOL]] %[[#B2V_ARG]] 6
; CHECK:   OpShiftLeftLogical %[[#I8]] %{{.*}} %[[#C6]]
; CHECK:   %[[#B2V_OR6:]] = OpBitwiseOr %[[#I8]]
;
; CHECK:   %[[#B2V_E7:]] = OpCompositeExtract %[[#BOOL]] %[[#B2V_ARG]] 7
; CHECK:   %[[#B2V_S7:]] = OpSelect %[[#I8]] %[[#B2V_E7]] %[[#ONE]] %[[#ZERO]]
; CHECK:   %[[#B2V_SHL7:]] = OpShiftLeftLogical %[[#I8]] %[[#B2V_S7]] %[[#C7]]
; CHECK:   %[[#B2V_OR7:]] = OpBitwiseOr %[[#I8]] %[[#B2V_OR6]] %[[#B2V_SHL7]]
; CHECK:   %[[#BC_TO_VEC1:]] = OpBitcast %[[#VEC1_I8]] %[[#B2V_OR7]]
; CHECK:   OpReturnValue %[[#BC_TO_VEC1]]
define <1 x i8> @boolvec_to_vec(<8 x i1> %v) {
  %r = bitcast <8 x i1> %v to <1 x i8>
  ret <1 x i8> %r
}

; bitcast <1 x i8> to <8 x i1>
;
; CHECK:   %[[#V2B:]] = OpFunction %[[#BVEC8]]
; CHECK:   %[[#V2B_ARG:]] = OpFunctionParameter %[[#VEC1_I8]]
; CHECK:   OpLabel
;
; CHECK:   %[[#BC_VEC1_INT8_TO_I8:]] = OpBitcast %[[#I8]] %[[#V2B_ARG]]
; CHECK:   %[[#V2B_A0:]] = OpBitwiseAnd %[[#I8]] %[[#BC_VEC1_INT8_TO_I8]] %[[#ONE]]
; CHECK:   %[[#V2B_C0:]] = OpINotEqual %[[#BOOL]] %[[#V2B_A0]] %[[#ZERO]]
; CHECK:   %[[#V2B_I0:]] = OpCompositeInsert %[[#BVEC8]] %[[#V2B_C0]] %{{.*}} 0
;
; CHECK:   %[[#V2B_A1:]] = OpBitwiseAnd %[[#I8]] %[[#BC_VEC1_INT8_TO_I8]] %[[#TWO]]
; CHECK:   %[[#V2B_C1:]] = OpINotEqual %[[#BOOL]] %[[#V2B_A1]] %[[#ZERO]]
; CHECK:   %[[#V2B_I1:]] = OpCompositeInsert %[[#BVEC8]] %[[#V2B_C1]] %[[#V2B_I0]] 1
;
; CHECK:   %[[#V2B_A2:]] = OpBitwiseAnd %[[#I8]] %[[#BC_VEC1_INT8_TO_I8]] %[[#FOUR]]
; CHECK:   %[[#V2B_C2:]] = OpINotEqual %[[#BOOL]] %[[#V2B_A2]] %[[#ZERO]]
; CHECK:   %[[#V2B_I2:]] = OpCompositeInsert %[[#BVEC8]] %[[#V2B_C2]] %[[#V2B_I1]] 2
;
; CHECK:   OpBitwiseAnd %[[#I8]] %[[#BC_VEC1_INT8_TO_I8]] %[[#EIGHT]]
; CHECK:   OpINotEqual %[[#BOOL]]
; CHECK:   %[[#V2B_I3:]] = OpCompositeInsert %[[#BVEC8]] %{{.*}} %[[#V2B_I2]] 3
; CHECK:   OpBitwiseAnd %[[#I8]] %[[#BC_VEC1_INT8_TO_I8]] %[[#C16]]
; CHECK:   OpINotEqual %[[#BOOL]]
; CHECK:   %[[#V2B_I4:]] = OpCompositeInsert %[[#BVEC8]] %{{.*}} %[[#V2B_I3]] 4
; CHECK:   OpBitwiseAnd %[[#I8]] %[[#BC_VEC1_INT8_TO_I8]] %[[#C32]]
; CHECK:   OpINotEqual %[[#BOOL]]
; CHECK:   %[[#V2B_I5:]] = OpCompositeInsert %[[#BVEC8]] %{{.*}} %[[#V2B_I4]] 5
; CHECK:   OpBitwiseAnd %[[#I8]] %[[#BC_VEC1_INT8_TO_I8]] %[[#C64]]
; CHECK:   OpINotEqual %[[#BOOL]]
; CHECK:   %[[#V2B_I6:]] = OpCompositeInsert %[[#BVEC8]] %{{.*}} %[[#V2B_I5]] 6
;
; CHECK:   %[[#V2B_A7:]] = OpBitwiseAnd %[[#I8]] %[[#BC_VEC1_INT8_TO_I8]] %[[#C128]]
; CHECK:   %[[#V2B_C7:]] = OpINotEqual %[[#BOOL]] %[[#V2B_A7]] %[[#ZERO]]
; CHECK:   %[[#V2B_I7:]] = OpCompositeInsert %[[#BVEC8]] %[[#V2B_C7]] %[[#V2B_I6]] 7
;
; CHECK:   OpReturnValue %[[#V2B_I77:]]
define <8 x i1> @vec_to_boolvec(<1 x i8> %v) {
  %r = bitcast <1 x i8> %v to <8 x i1>
  ret <8 x i1> %r
}

; bitcast i8 to <8 x i1>
; Tests each bit with AND + INotEqual, inserts each bool into the result vector.
;
; CHECK:   %[[#S2B:]] = OpFunction %[[#BVEC8]]
; CHECK-SAME: -- Begin function scalar_to_boolvec
; CHECK:   %[[#S2B_ARG:]] = OpFunctionParameter %[[#I8]]
; CHECK:   OpLabel
;
; CHECK:   %[[#A0:]] = OpBitwiseAnd %[[#I8]] %[[#S2B_ARG]] %[[#ONE]]
; CHECK:   %[[#C0:]] = OpINotEqual %[[#BOOL]] %[[#A0]] %[[#ZERO]]
; CHECK:   %[[#I0:]] = OpCompositeInsert %[[#BVEC8]] %[[#C0]] %{{.*}} 0
;
; CHECK:   %[[#A1:]] = OpBitwiseAnd %[[#I8]] %[[#S2B_ARG]] %[[#TWO]]
; CHECK:   %[[#C1:]] = OpINotEqual %[[#BOOL]] %[[#A1]] %[[#ZERO]]
; CHECK:   %[[#I1:]] = OpCompositeInsert %[[#BVEC8]] %[[#C1]] %[[#I0]] 1
;
; CHECK:   %[[#A2:]] = OpBitwiseAnd %[[#I8]] %[[#S2B_ARG]] %[[#FOUR]]
; CHECK:   %[[#C2:]] = OpINotEqual %[[#BOOL]] %[[#A2]] %[[#ZERO]]
; CHECK:   %[[#I2:]] = OpCompositeInsert %[[#BVEC8]] %[[#C2]] %[[#I1]] 2
;
; CHECK:   OpBitwiseAnd %[[#I8]] %[[#S2B_ARG]] %[[#EIGHT]]
; CHECK:   OpINotEqual %[[#BOOL]]
; CHECK:   %[[#I3:]] = OpCompositeInsert %[[#BVEC8]] %{{.*}} %[[#I2]] 3
; CHECK:   OpBitwiseAnd %[[#I8]] %[[#S2B_ARG]] %[[#C16]]
; CHECK:   OpINotEqual %[[#BOOL]]
; CHECK:   %[[#I4:]] = OpCompositeInsert %[[#BVEC8]] %{{.*}} %[[#I3]] 4
; CHECK:   OpBitwiseAnd %[[#I8]] %[[#S2B_ARG]] %[[#C32]]
; CHECK:   OpINotEqual %[[#BOOL]]
; CHECK:   %[[#I5:]] = OpCompositeInsert %[[#BVEC8]] %{{.*}} %[[#I4]] 5
; CHECK:   OpBitwiseAnd %[[#I8]] %[[#S2B_ARG]] %[[#C64]]
; CHECK:   OpINotEqual %[[#BOOL]]
; CHECK:   %[[#I6:]] = OpCompositeInsert %[[#BVEC8]] %{{.*}} %[[#I5]] 6
;
; CHECK:   %[[#A7:]] = OpBitwiseAnd %[[#I8]] %[[#S2B_ARG]] %[[#C128]]
; CHECK:   %[[#C7B:]] = OpINotEqual %[[#BOOL]] %[[#A7]] %[[#ZERO]]
; CHECK:   %[[#I7:]] = OpCompositeInsert %[[#BVEC8]] %[[#C7B]] %[[#I6]] 7
;
; CHECK:   OpReturnValue %[[#I7]]
define <8 x i1> @scalar_to_boolvec(i8 %v) {
  %r = bitcast i8 %v to <8 x i1>
  ret <8 x i1> %r
}

; bitcast <17 x i1> to I17
;
; CHECK:   %[[#B2S17:]] = OpFunction %[[#I17]]
; CHECK:   %[[#B2S17_ARG:]] = OpFunctionParameter %[[#BVEC17]]
; CHECK:   OpLabel
;
; CHECK:   %[[#B2S17_E0:]] = OpCompositeExtract %[[#BOOL]] %[[#B2S17_ARG]] 0
; CHECK:   %[[#B2S17_S0:]] = OpSelect %[[#I17]] %[[#B2S17_E0]] %[[#ONE17]] %[[#ZERO17]]
; CHECK:   %[[#B2S17_OR0:]] = OpBitwiseOr %[[#I17]] %[[#ZERO17]] %[[#B2S17_S0]]
;
; CHECK:   %[[#B2S17_E1:]] = OpCompositeExtract %[[#BOOL]] %[[#B2S17_ARG]] 1
; CHECK:   %[[#B2S17_S1:]] = OpSelect %[[#I17]] %[[#B2S17_E1]] %[[#ONE17]] %[[#ZERO17]]
; CHECK:   %[[#B2S17_SHL1:]] = OpShiftLeftLogical %[[#I17]] %[[#B2S17_S1]] %[[#ONE17]]
; CHECK:   %[[#B2S17_OR1:]] = OpBitwiseOr %[[#I17]] %[[#B2S17_OR0]] %[[#B2S17_SHL1]]
;
; CHECK:   %[[#B2S17_E2:]] = OpCompositeExtract %[[#BOOL]] %[[#B2S17_ARG]] 2
; CHECK:   %[[#B2S17_S2:]] = OpSelect %[[#I17]] %[[#B2S17_E2]] %[[#ONE17]] %[[#ZERO17]]
; CHECK:   %[[#B2S17_SHL2:]] = OpShiftLeftLogical %[[#I17]] %[[#B2S17_S2]] %[[#TWO17]]
; CHECK:   %[[#B2S17_OR2:]] = OpBitwiseOr %[[#I17]] %[[#B2S17_OR1]] %[[#B2S17_SHL2]]
;
; CHECK:   OpCompositeExtract %[[#BOOL]] %[[#B2S17_ARG]] 3
; CHECK:   OpShiftLeftLogical %[[#I17]] %{{.*}} %[[#C3_17]]
; CHECK:   %[[#B2S17_OR3:]] = OpBitwiseOr %[[#I17]]
; CHECK:   OpCompositeExtract %[[#BOOL]] %[[#B2S17_ARG]] 4
; CHECK:   OpShiftLeftLogical %[[#I17]] %{{.*}} %[[#C4_17]]
; CHECK:   %[[#B2S17_OR4:]] = OpBitwiseOr %[[#I17]]
; CHECK:   OpCompositeExtract %[[#BOOL]] %[[#B2S17_ARG]] 5
; CHECK:   OpShiftLeftLogical %[[#I17]] %{{.*}} %[[#C5_17]]
; CHECK:   %[[#B2S17_OR5:]] = OpBitwiseOr %[[#I17]]
; CHECK:   OpCompositeExtract %[[#BOOL]] %[[#B2S17_ARG]] 6
; CHECK:   OpShiftLeftLogical %[[#I17]] %{{.*}} %[[#C6_17]]
; CHECK:   %[[#B2S17_OR6:]] = OpBitwiseOr %[[#I17]]
; CHECK:   OpCompositeExtract %[[#BOOL]] %[[#B2S17_ARG]] 7
; CHECK:   OpShiftLeftLogical %[[#I17]] %{{.*}} %[[#C7_17]]
; CHECK:   %[[#B2S17_OR7:]] = OpBitwiseOr %[[#I17]]
; CHECK:   OpCompositeExtract %[[#BOOL]] %[[#B2S17_ARG]] 8
; CHECK:   OpShiftLeftLogical %[[#I17]] %{{.*}} %[[#C8_17]]
; CHECK:   %[[#B2S17_OR8:]] = OpBitwiseOr %[[#I17]]
; CHECK:   OpCompositeExtract %[[#BOOL]] %[[#B2S17_ARG]] 9
; CHECK:   OpShiftLeftLogical %[[#I17]] %{{.*}} %[[#C9_17]]
; CHECK:   %[[#B2S17_OR9:]] = OpBitwiseOr %[[#I17]]
; CHECK:   OpCompositeExtract %[[#BOOL]] %[[#B2S17_ARG]] 10
; CHECK:   OpShiftLeftLogical %[[#I17]] %{{.*}} %[[#C10_17]]
; CHECK:   %[[#B2S17_OR10:]] = OpBitwiseOr %[[#I17]]
; CHECK:   OpCompositeExtract %[[#BOOL]] %[[#B2S17_ARG]] 11
; CHECK:   OpShiftLeftLogical %[[#I17]] %{{.*}} %[[#C11_17]]
; CHECK:   %[[#B2S17_OR11:]] = OpBitwiseOr %[[#I17]]
; CHECK:   OpCompositeExtract %[[#BOOL]] %[[#B2S17_ARG]] 12
; CHECK:   OpShiftLeftLogical %[[#I17]] %{{.*}} %[[#C12_17]]
; CHECK:   %[[#B2S17_OR12:]] = OpBitwiseOr %[[#I17]]
; CHECK:   OpCompositeExtract %[[#BOOL]] %[[#B2S17_ARG]] 13
; CHECK:   OpShiftLeftLogical %[[#I17]] %{{.*}} %[[#C13_17]]
; CHECK:   %[[#B2S17_OR13:]] = OpBitwiseOr %[[#I17]]
; CHECK:   OpCompositeExtract %[[#BOOL]] %[[#B2S17_ARG]] 14
; CHECK:   OpShiftLeftLogical %[[#I17]] %{{.*}} %[[#C14_17]]
; CHECK:   %[[#B2S17_OR14:]] = OpBitwiseOr %[[#I17]]
; CHECK:   %[[#B2S17_E15:]] = OpCompositeExtract %[[#BOOL]] %[[#B2S17_ARG]] 15
; CHECK:   %[[#B2S17_S15:]] = OpSelect %[[#I17]] %[[#B2S17_E15]] %[[#ONE17]] %[[#ZERO17]]
; CHECK:   %[[#B2S17_SHL15:]] = OpShiftLeftLogical %[[#I17]] %[[#B2S17_S15]] %[[#C15_17]]
; CHECK:   %[[#B2S17_OR15:]] = OpBitwiseOr %[[#I17]] %[[#B2S17_OR14]] %[[#B2S17_SHL15]]
;
; CHECK:   %[[#B2S17_E16:]] = OpCompositeExtract %[[#BOOL]] %[[#B2S17_ARG]] 16
; CHECK:   %[[#B2S17_S16:]] = OpSelect %[[#I17]] %[[#B2S17_E16]] %[[#ONE17]] %[[#ZERO17]]
; CHECK:   %[[#B2S17_SHL16:]] = OpShiftLeftLogical %[[#I17]] %[[#B2S17_S16]] %[[#C16_17]]
; CHECK:   %[[#B2S17_OR16:]] = OpBitwiseOr %[[#I17]] %[[#B2S17_OR15]] %[[#B2S17_SHL16]]
;
; CHECK:   OpReturnValue %[[#B2S17_OR16]]
define i17 @boolvec17_to_scalar(<17 x i1> %v) {
  %r = bitcast <17 x i1> %v to i17
  ret i17 %r
}

; bitcast i17 to <17 x i1>
;
; CHECK:   %[[#S2B17:]] = OpFunction %[[#BVEC17]]
; CHECK:   %[[#S2B17_ARG:]] = OpFunctionParameter %[[#I17]]
; CHECK:   OpLabel
;
; CHECK:   %[[#S2B17_A0:]] = OpBitwiseAnd %[[#I17]] %[[#S2B17_ARG]] %[[#ONE17]]
; CHECK:   %[[#S2B17_C0:]] = OpINotEqual %[[#BOOL]] %[[#S2B17_A0]] %[[#ZERO17]]
; CHECK:   %[[#S2B17_I0:]] = OpCompositeInsert %[[#BVEC17]] %[[#S2B17_C0]] %{{.*}} 0
;
; CHECK:   %[[#S2B17_A1:]] = OpBitwiseAnd %[[#I17]] %[[#S2B17_ARG]] %[[#TWO17]]
; CHECK:   %[[#S2B17_C1:]] = OpINotEqual %[[#BOOL]] %[[#S2B17_A1]] %[[#ZERO17]]
; CHECK:   %[[#S2B17_I1:]] = OpCompositeInsert %[[#BVEC17]] %[[#S2B17_C1]] %[[#S2B17_I0]] 1
;
; CHECK:   %[[#S2B17_A2:]] = OpBitwiseAnd %[[#I17]] %[[#S2B17_ARG]] %[[#C4_17]]
; CHECK:   %[[#S2B17_C2:]] = OpINotEqual %[[#BOOL]] %[[#S2B17_A2]] %[[#ZERO17]]
; CHECK:   %[[#S2B17_I2:]] = OpCompositeInsert %[[#BVEC17]] %[[#S2B17_C2]] %[[#S2B17_I1]] 2
;
; CHECK:   OpBitwiseAnd %[[#I17]] %[[#S2B17_ARG]] %[[#C8_17]]
; CHECK:   OpINotEqual %[[#BOOL]]
; CHECK:   %[[#S2B17_I3:]] = OpCompositeInsert %[[#BVEC17]] %{{.*}} %[[#S2B17_I2]] 3
; CHECK:   OpBitwiseAnd %[[#I17]] %[[#S2B17_ARG]] %[[#C16_17]]
; CHECK:   OpINotEqual %[[#BOOL]]
; CHECK:   %[[#S2B17_I4:]] = OpCompositeInsert %[[#BVEC17]] %{{.*}} %[[#S2B17_I3]] 4
; CHECK:   OpBitwiseAnd %[[#I17]] %[[#S2B17_ARG]] %[[#C32_17]]
; CHECK:   OpINotEqual %[[#BOOL]]
; CHECK:   %[[#S2B17_I5:]] = OpCompositeInsert %[[#BVEC17]] %{{.*}} %[[#S2B17_I4]] 5
; CHECK:   OpBitwiseAnd %[[#I17]] %[[#S2B17_ARG]] %[[#C64_17]]
; CHECK:   OpINotEqual %[[#BOOL]]
; CHECK:   %[[#S2B17_I6:]] = OpCompositeInsert %[[#BVEC17]] %{{.*}} %[[#S2B17_I5]] 6
; CHECK:   OpBitwiseAnd %[[#I17]] %[[#S2B17_ARG]] %[[#C128_17]]
; CHECK:   OpINotEqual %[[#BOOL]]
; CHECK:   %[[#S2B17_I7:]] = OpCompositeInsert %[[#BVEC17]] %{{.*}} %[[#S2B17_I6]] 7
; CHECK:   OpBitwiseAnd %[[#I17]] %[[#S2B17_ARG]] %[[#C256_17]]
; CHECK:   OpINotEqual %[[#BOOL]]
; CHECK:   %[[#S2B17_I8:]] = OpCompositeInsert %[[#BVEC17]] %{{.*}} %[[#S2B17_I7]] 8
; CHECK:   OpBitwiseAnd %[[#I17]] %[[#S2B17_ARG]] %[[#C512_17]]
; CHECK:   OpINotEqual %[[#BOOL]]
; CHECK:   %[[#S2B17_I9:]] = OpCompositeInsert %[[#BVEC17]] %{{.*}} %[[#S2B17_I8]] 9
; CHECK:   OpBitwiseAnd %[[#I17]] %[[#S2B17_ARG]] %[[#C1024_17]]
; CHECK:   OpINotEqual %[[#BOOL]]
; CHECK:   %[[#S2B17_I10:]] = OpCompositeInsert %[[#BVEC17]] %{{.*}} %[[#S2B17_I9]] 10
; CHECK:   OpBitwiseAnd %[[#I17]] %[[#S2B17_ARG]] %[[#C2048_17]]
; CHECK:   OpINotEqual %[[#BOOL]]
; CHECK:   %[[#S2B17_I11:]] = OpCompositeInsert %[[#BVEC17]] %{{.*}} %[[#S2B17_I10]] 11
; CHECK:   OpBitwiseAnd %[[#I17]] %[[#S2B17_ARG]] %[[#C4096_17]]
; CHECK:   OpINotEqual %[[#BOOL]]
; CHECK:   %[[#S2B17_I12:]] = OpCompositeInsert %[[#BVEC17]] %{{.*}} %[[#S2B17_I11]] 12
; CHECK:   OpBitwiseAnd %[[#I17]] %[[#S2B17_ARG]] %[[#C8192_17]]
; CHECK:   OpINotEqual %[[#BOOL]]
; CHECK:   %[[#S2B17_I13:]] = OpCompositeInsert %[[#BVEC17]] %{{.*}} %[[#S2B17_I12]] 13
; CHECK:   OpBitwiseAnd %[[#I17]] %[[#S2B17_ARG]] %[[#C16384_17]]
; CHECK:   OpINotEqual %[[#BOOL]]
; CHECK:   %[[#S2B17_I14:]] = OpCompositeInsert %[[#BVEC17]] %{{.*}} %[[#S2B17_I13]] 14
; CHECK:   %[[#S2B17_A15:]] = OpBitwiseAnd %[[#I17]] %[[#S2B17_ARG]] %[[#C32768_17]]
; CHECK:   %[[#S2B17_C15:]] = OpINotEqual %[[#BOOL]] %[[#S2B17_A15]] %[[#ZERO17]]
; CHECK:   %[[#S2B17_I15:]] = OpCompositeInsert %[[#BVEC17]] %[[#S2B17_C15]] %[[#S2B17_I14]] 15
;
; CHECK:   %[[#S2B17_A16:]] = OpBitwiseAnd %[[#I17]] %[[#S2B17_ARG]] %[[#C65536_17]]
; CHECK:   %[[#S2B17_C16:]] = OpINotEqual %[[#BOOL]] %[[#S2B17_A16]] %[[#ZERO17]]
; CHECK:   %[[#S2B17_I16:]] = OpCompositeInsert %[[#BVEC17]] %[[#S2B17_C16]] %[[#S2B17_I15]] 16
;
; CHECK:   OpReturnValue %[[#S2B17_I16]]
define <17 x i1> @scalar_to_boolvec17(i17 %v) {
  %r = bitcast i17 %v to <17 x i1>
  ret <17 x i1> %r
}
