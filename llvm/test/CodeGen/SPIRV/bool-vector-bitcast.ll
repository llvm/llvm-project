; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; FIXME: enabled on Vulkan env, when legalization of vectors > 4 elements is
; fully supported.

; Verify that bitcasts between bool vectors and other types are decomposed
; into element-wise operations instead of generating OpBitcast, which is
; invalid for OpTypeBool.

; CHECK-DAG: %[[#BOOL:]] = OpTypeBool
; CHECK-DAG: %[[#BVEC8:]] = OpTypeVector %[[#BOOL]] 8
; CHECK-DAG: %[[#BVEC16:]] = OpTypeVector %[[#BOOL]] 16
; CHECK-DAG: %[[#I8:]] = OpTypeInt 8 0
; CHECK-DAG: %[[#I16:]] = OpTypeInt 16 0

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

; CHECK-DAG: %[[#ZERO16:]] = OpConstantNull %[[#I16]]
; CHECK-DAG: %[[#ONE16:]] = OpConstant %[[#I16]] 1{{$}}
; CHECK-DAG: %[[#TWO16:]] = OpConstant %[[#I16]] 2{{$}}
; CHECK-DAG: %[[#C3_16:]] = OpConstant %[[#I16]] 3{{$}}
; CHECK-DAG: %[[#C4_16:]] = OpConstant %[[#I16]] 4{{$}}
; CHECK-DAG: %[[#C5_16:]] = OpConstant %[[#I16]] 5{{$}}
; CHECK-DAG: %[[#C6_16:]] = OpConstant %[[#I16]] 6{{$}}
; CHECK-DAG: %[[#C7_16:]] = OpConstant %[[#I16]] 7{{$}}
; CHECK-DAG: %[[#C8_16:]] = OpConstant %[[#I16]] 8{{$}}
; CHECK-DAG: %[[#C9_16:]] = OpConstant %[[#I16]] 9{{$}}
; CHECK-DAG: %[[#C10_16:]] = OpConstant %[[#I16]] 10{{$}}
; CHECK-DAG: %[[#C11_16:]] = OpConstant %[[#I16]] 11{{$}}
; CHECK-DAG: %[[#C12_16:]] = OpConstant %[[#I16]] 12{{$}}
; CHECK-DAG: %[[#C13_16:]] = OpConstant %[[#I16]] 13{{$}}
; CHECK-DAG: %[[#C14_16:]] = OpConstant %[[#I16]] 14{{$}}
; CHECK-DAG: %[[#C15_16:]] = OpConstant %[[#I16]] 15{{$}}
; CHECK-DAG: %[[#C16_16:]] = OpConstant %[[#I16]] 16{{$}}
; CHECK-DAG: %[[#C32_16:]] = OpConstant %[[#I16]] 32{{$}}
; CHECK-DAG: %[[#C64_16:]] = OpConstant %[[#I16]] 64{{$}}
; CHECK-DAG: %[[#C128_16:]] = OpConstant %[[#I16]] 128{{$}}
; CHECK-DAG: %[[#C256_16:]] = OpConstant %[[#I16]] 256{{$}}
; CHECK-DAG: %[[#C512_16:]] = OpConstant %[[#I16]] 512{{$}}
; CHECK-DAG: %[[#C1024_16:]] = OpConstant %[[#I16]] 1024{{$}}
; CHECK-DAG: %[[#C2048_16:]] = OpConstant %[[#I16]] 2048{{$}}
; CHECK-DAG: %[[#C4096_16:]] = OpConstant %[[#I16]] 4096{{$}}
; CHECK-DAG: %[[#C8192_16:]] = OpConstant %[[#I16]] 8192{{$}}
; CHECK-DAG: %[[#C16384_16:]] = OpConstant %[[#I16]] 16384{{$}}
; CHECK-DAG: %[[#C32768_16:]] = OpConstant %[[#I16]] 32768{{$}}


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
; CHECK:   %[[#B2V:]] = OpFunction %[[#I8]]
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
;
; CHECK:   OpReturnValue %[[#B2V_OR7]]
define <1 x i8> @boolvec_to_vec(<8 x i1> %v) {
  %r = bitcast <8 x i1> %v to <1 x i8>
  ret <1 x i8> %r
}

; bitcast <1 x i8> to <8 x i1>
;
; CHECK:   %[[#V2B:]] = OpFunction %[[#BVEC8]]
; CHECK:   %[[#V2B_ARG:]] = OpFunctionParameter %[[#I8]]
; CHECK:   OpLabel
;
; CHECK:   %[[#V2B_A0:]] = OpBitwiseAnd %[[#I8]] %[[#V2B_ARG]] %[[#ONE]]
; CHECK:   %[[#V2B_C0:]] = OpINotEqual %[[#BOOL]] %[[#V2B_A0]] %[[#ZERO]]
; CHECK:   %[[#V2B_I0:]] = OpCompositeInsert %[[#BVEC8]] %[[#V2B_C0]] %{{.*}} 0
;
; CHECK:   %[[#V2B_A1:]] = OpBitwiseAnd %[[#I8]] %[[#V2B_ARG]] %[[#TWO]]
; CHECK:   %[[#V2B_C1:]] = OpINotEqual %[[#BOOL]] %[[#V2B_A1]] %[[#ZERO]]
; CHECK:   %[[#V2B_I1:]] = OpCompositeInsert %[[#BVEC8]] %[[#V2B_C1]] %[[#V2B_I0]] 1
;
; CHECK:   %[[#V2B_A2:]] = OpBitwiseAnd %[[#I8]] %[[#V2B_ARG]] %[[#FOUR]]
; CHECK:   %[[#V2B_C2:]] = OpINotEqual %[[#BOOL]] %[[#V2B_A2]] %[[#ZERO]]
; CHECK:   %[[#V2B_I2:]] = OpCompositeInsert %[[#BVEC8]] %[[#V2B_C2]] %[[#V2B_I1]] 2
;
; CHECK:   OpBitwiseAnd %[[#I8]] %[[#V2B_ARG]] %[[#EIGHT]]
; CHECK:   OpINotEqual %[[#BOOL]]
; CHECK:   %[[#V2B_I3:]] = OpCompositeInsert %[[#BVEC8]] %{{.*}} %[[#V2B_I2]] 3
; CHECK:   OpBitwiseAnd %[[#I8]] %[[#V2B_ARG]] %[[#C16]]
; CHECK:   OpINotEqual %[[#BOOL]]
; CHECK:   %[[#V2B_I4:]] = OpCompositeInsert %[[#BVEC8]] %{{.*}} %[[#V2B_I3]] 4
; CHECK:   OpBitwiseAnd %[[#I8]] %[[#V2B_ARG]] %[[#C32]]
; CHECK:   OpINotEqual %[[#BOOL]]
; CHECK:   %[[#V2B_I5:]] = OpCompositeInsert %[[#BVEC8]] %{{.*}} %[[#V2B_I4]] 5
; CHECK:   OpBitwiseAnd %[[#I8]] %[[#V2B_ARG]] %[[#C64]]
; CHECK:   OpINotEqual %[[#BOOL]]
; CHECK:   %[[#V2B_I6:]] = OpCompositeInsert %[[#BVEC8]] %{{.*}} %[[#V2B_I5]] 6
;
; CHECK:   %[[#V2B_A7:]] = OpBitwiseAnd %[[#I8]] %[[#V2B_ARG]] %[[#C128]]
; CHECK:   %[[#V2B_C7:]] = OpINotEqual %[[#BOOL]] %[[#V2B_A7]] %[[#ZERO]]
; CHECK:   %[[#V2B_I7:]] = OpCompositeInsert %[[#BVEC8]] %[[#V2B_C7]] %[[#V2B_I6]] 7
;
; CHECK:   OpReturnValue %[[#V2B_I7]]
define <8 x i1> @vec_to_boolvec(<1 x i8> %v) {
  %r = bitcast <1 x i8> %v to <8 x i1>
  ret <8 x i1> %r
}

; bitcast i8 to <8 x i1>
; Tests each bit with AND + INotEqual, inserts each bool into the result vector.
;
; CHECK:   %[[#S2B:]] = OpFunction %[[#BVEC8]]
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

; bitcast <16 x i1> to i16
;
; CHECK:   %[[#B2S16:]] = OpFunction %[[#I16]]
; CHECK:   %[[#B2S16_ARG:]] = OpFunctionParameter %[[#BVEC16]]
; CHECK:   OpLabel
;
; CHECK:   %[[#B2S16_E0:]] = OpCompositeExtract %[[#BOOL]] %[[#B2S16_ARG]] 0
; CHECK:   %[[#B2S16_S0:]] = OpSelect %[[#I16]] %[[#B2S16_E0]] %[[#ONE16]] %[[#ZERO16]]
; CHECK:   %[[#B2S16_OR0:]] = OpBitwiseOr %[[#I16]] %[[#ZERO16]] %[[#B2S16_S0]]
;
; CHECK:   %[[#B2S16_E1:]] = OpCompositeExtract %[[#BOOL]] %[[#B2S16_ARG]] 1
; CHECK:   %[[#B2S16_S1:]] = OpSelect %[[#I16]] %[[#B2S16_E1]] %[[#ONE16]] %[[#ZERO16]]
; CHECK:   %[[#B2S16_SHL1:]] = OpShiftLeftLogical %[[#I16]] %[[#B2S16_S1]] %[[#ONE16]]
; CHECK:   %[[#B2S16_OR1:]] = OpBitwiseOr %[[#I16]] %[[#B2S16_OR0]] %[[#B2S16_SHL1]]
;
; CHECK:   %[[#B2S16_E2:]] = OpCompositeExtract %[[#BOOL]] %[[#B2S16_ARG]] 2
; CHECK:   %[[#B2S16_S2:]] = OpSelect %[[#I16]] %[[#B2S16_E2]] %[[#ONE16]] %[[#ZERO16]]
; CHECK:   %[[#B2S16_SHL2:]] = OpShiftLeftLogical %[[#I16]] %[[#B2S16_S2]] %[[#TWO16]]
; CHECK:   %[[#B2S16_OR2:]] = OpBitwiseOr %[[#I16]] %[[#B2S16_OR1]] %[[#B2S16_SHL2]]
;
; CHECK:   OpCompositeExtract %[[#BOOL]] %[[#B2S16_ARG]] 3
; CHECK:   OpShiftLeftLogical %[[#I16]] %{{.*}} %[[#C3_16]]
; CHECK:   %[[#B2S16_OR3:]] = OpBitwiseOr %[[#I16]]
; CHECK:   OpCompositeExtract %[[#BOOL]] %[[#B2S16_ARG]] 4
; CHECK:   OpShiftLeftLogical %[[#I16]] %{{.*}} %[[#C4_16]]
; CHECK:   %[[#B2S16_OR4:]] = OpBitwiseOr %[[#I16]]
; CHECK:   OpCompositeExtract %[[#BOOL]] %[[#B2S16_ARG]] 5
; CHECK:   OpShiftLeftLogical %[[#I16]] %{{.*}} %[[#C5_16]]
; CHECK:   %[[#B2S16_OR5:]] = OpBitwiseOr %[[#I16]]
; CHECK:   OpCompositeExtract %[[#BOOL]] %[[#B2S16_ARG]] 6
; CHECK:   OpShiftLeftLogical %[[#I16]] %{{.*}} %[[#C6_16]]
; CHECK:   %[[#B2S16_OR6:]] = OpBitwiseOr %[[#I16]]
; CHECK:   OpCompositeExtract %[[#BOOL]] %[[#B2S16_ARG]] 7
; CHECK:   OpShiftLeftLogical %[[#I16]] %{{.*}} %[[#C7_16]]
; CHECK:   %[[#B2S16_OR7:]] = OpBitwiseOr %[[#I16]]
; CHECK:   OpCompositeExtract %[[#BOOL]] %[[#B2S16_ARG]] 8
; CHECK:   OpShiftLeftLogical %[[#I16]] %{{.*}} %[[#C8_16]]
; CHECK:   %[[#B2S16_OR8:]] = OpBitwiseOr %[[#I16]]
; CHECK:   OpCompositeExtract %[[#BOOL]] %[[#B2S16_ARG]] 9
; CHECK:   OpShiftLeftLogical %[[#I16]] %{{.*}} %[[#C9_16]]
; CHECK:   %[[#B2S16_OR9:]] = OpBitwiseOr %[[#I16]]
; CHECK:   OpCompositeExtract %[[#BOOL]] %[[#B2S16_ARG]] 10
; CHECK:   OpShiftLeftLogical %[[#I16]] %{{.*}} %[[#C10_16]]
; CHECK:   %[[#B2S16_OR10:]] = OpBitwiseOr %[[#I16]]
; CHECK:   OpCompositeExtract %[[#BOOL]] %[[#B2S16_ARG]] 11
; CHECK:   OpShiftLeftLogical %[[#I16]] %{{.*}} %[[#C11_16]]
; CHECK:   %[[#B2S16_OR11:]] = OpBitwiseOr %[[#I16]]
; CHECK:   OpCompositeExtract %[[#BOOL]] %[[#B2S16_ARG]] 12
; CHECK:   OpShiftLeftLogical %[[#I16]] %{{.*}} %[[#C12_16]]
; CHECK:   %[[#B2S16_OR12:]] = OpBitwiseOr %[[#I16]]
; CHECK:   OpCompositeExtract %[[#BOOL]] %[[#B2S16_ARG]] 13
; CHECK:   OpShiftLeftLogical %[[#I16]] %{{.*}} %[[#C13_16]]
; CHECK:   %[[#B2S16_OR13:]] = OpBitwiseOr %[[#I16]]
; CHECK:   OpCompositeExtract %[[#BOOL]] %[[#B2S16_ARG]] 14
; CHECK:   OpShiftLeftLogical %[[#I16]] %{{.*}} %[[#C14_16]]
; CHECK:   %[[#B2S16_OR14:]] = OpBitwiseOr %[[#I16]]
;
; CHECK:   %[[#B2S16_E15:]] = OpCompositeExtract %[[#BOOL]] %[[#B2S16_ARG]] 15
; CHECK:   %[[#B2S16_S15:]] = OpSelect %[[#I16]] %[[#B2S16_E15]] %[[#ONE16]] %[[#ZERO16]]
; CHECK:   %[[#B2S16_SHL15:]] = OpShiftLeftLogical %[[#I16]] %[[#B2S16_S15]] %[[#C15_16]]
; CHECK:   %[[#B2S16_OR15:]] = OpBitwiseOr %[[#I16]] %[[#B2S16_OR14]] %[[#B2S16_SHL15]]
;
; CHECK:   OpReturnValue %[[#B2S16_OR15]]
define i16 @boolvec16_to_scalar(<16 x i1> %v) {
  %r = bitcast <16 x i1> %v to i16
  ret i16 %r
}

; bitcast i16 to <16 x i1>
;
; CHECK:   %[[#S2B16:]] = OpFunction %[[#BVEC16]]
; CHECK:   %[[#S2B16_ARG:]] = OpFunctionParameter %[[#I16]]
; CHECK:   OpLabel
;
; CHECK:   %[[#S2B16_A0:]] = OpBitwiseAnd %[[#I16]] %[[#S2B16_ARG]] %[[#ONE16]]
; CHECK:   %[[#S2B16_C0:]] = OpINotEqual %[[#BOOL]] %[[#S2B16_A0]] %[[#ZERO16]]
; CHECK:   %[[#S2B16_I0:]] = OpCompositeInsert %[[#BVEC16]] %[[#S2B16_C0]] %{{.*}} 0
;
; CHECK:   %[[#S2B16_A1:]] = OpBitwiseAnd %[[#I16]] %[[#S2B16_ARG]] %[[#TWO16]]
; CHECK:   %[[#S2B16_C1:]] = OpINotEqual %[[#BOOL]] %[[#S2B16_A1]] %[[#ZERO16]]
; CHECK:   %[[#S2B16_I1:]] = OpCompositeInsert %[[#BVEC16]] %[[#S2B16_C1]] %[[#S2B16_I0]] 1
;
; CHECK:   %[[#S2B16_A2:]] = OpBitwiseAnd %[[#I16]] %[[#S2B16_ARG]] %[[#C4_16]]
; CHECK:   %[[#S2B16_C2:]] = OpINotEqual %[[#BOOL]] %[[#S2B16_A2]] %[[#ZERO16]]
; CHECK:   %[[#S2B16_I2:]] = OpCompositeInsert %[[#BVEC16]] %[[#S2B16_C2]] %[[#S2B16_I1]] 2
;
; CHECK:   OpBitwiseAnd %[[#I16]] %[[#S2B16_ARG]] %[[#C8_16]]
; CHECK:   OpINotEqual %[[#BOOL]]
; CHECK:   %[[#S2B16_I3:]] = OpCompositeInsert %[[#BVEC16]] %{{.*}} %[[#S2B16_I2]] 3
; CHECK:   OpBitwiseAnd %[[#I16]] %[[#S2B16_ARG]] %[[#C16_16]]
; CHECK:   OpINotEqual %[[#BOOL]]
; CHECK:   %[[#S2B16_I4:]] = OpCompositeInsert %[[#BVEC16]] %{{.*}} %[[#S2B16_I3]] 4
; CHECK:   OpBitwiseAnd %[[#I16]] %[[#S2B16_ARG]] %[[#C32_16]]
; CHECK:   OpINotEqual %[[#BOOL]]
; CHECK:   %[[#S2B16_I5:]] = OpCompositeInsert %[[#BVEC16]] %{{.*}} %[[#S2B16_I4]] 5
; CHECK:   OpBitwiseAnd %[[#I16]] %[[#S2B16_ARG]] %[[#C64_16]]
; CHECK:   OpINotEqual %[[#BOOL]]
; CHECK:   %[[#S2B16_I6:]] = OpCompositeInsert %[[#BVEC16]] %{{.*}} %[[#S2B16_I5]] 6
; CHECK:   OpBitwiseAnd %[[#I16]] %[[#S2B16_ARG]] %[[#C128_16]]
; CHECK:   OpINotEqual %[[#BOOL]]
; CHECK:   %[[#S2B16_I7:]] = OpCompositeInsert %[[#BVEC16]] %{{.*}} %[[#S2B16_I6]] 7
; CHECK:   OpBitwiseAnd %[[#I16]] %[[#S2B16_ARG]] %[[#C256_16]]
; CHECK:   OpINotEqual %[[#BOOL]]
; CHECK:   %[[#S2B16_I8:]] = OpCompositeInsert %[[#BVEC16]] %{{.*}} %[[#S2B16_I7]] 8
; CHECK:   OpBitwiseAnd %[[#I16]] %[[#S2B16_ARG]] %[[#C512_16]]
; CHECK:   OpINotEqual %[[#BOOL]]
; CHECK:   %[[#S2B16_I9:]] = OpCompositeInsert %[[#BVEC16]] %{{.*}} %[[#S2B16_I8]] 9
; CHECK:   OpBitwiseAnd %[[#I16]] %[[#S2B16_ARG]] %[[#C1024_16]]
; CHECK:   OpINotEqual %[[#BOOL]]
; CHECK:   %[[#S2B16_I10:]] = OpCompositeInsert %[[#BVEC16]] %{{.*}} %[[#S2B16_I9]] 10
; CHECK:   OpBitwiseAnd %[[#I16]] %[[#S2B16_ARG]] %[[#C2048_16]]
; CHECK:   OpINotEqual %[[#BOOL]]
; CHECK:   %[[#S2B16_I11:]] = OpCompositeInsert %[[#BVEC16]] %{{.*}} %[[#S2B16_I10]] 11
; CHECK:   OpBitwiseAnd %[[#I16]] %[[#S2B16_ARG]] %[[#C4096_16]]
; CHECK:   OpINotEqual %[[#BOOL]]
; CHECK:   %[[#S2B16_I12:]] = OpCompositeInsert %[[#BVEC16]] %{{.*}} %[[#S2B16_I11]] 12
; CHECK:   OpBitwiseAnd %[[#I16]] %[[#S2B16_ARG]] %[[#C8192_16]]
; CHECK:   OpINotEqual %[[#BOOL]]
; CHECK:   %[[#S2B16_I13:]] = OpCompositeInsert %[[#BVEC16]] %{{.*}} %[[#S2B16_I12]] 13
; CHECK:   OpBitwiseAnd %[[#I16]] %[[#S2B16_ARG]] %[[#C16384_16]]
; CHECK:   OpINotEqual %[[#BOOL]]
; CHECK:   %[[#S2B16_I14:]] = OpCompositeInsert %[[#BVEC16]] %{{.*}} %[[#S2B16_I13]] 14
;
; CHECK:   %[[#S2B16_A15:]] = OpBitwiseAnd %[[#I16]] %[[#S2B16_ARG]] %[[#C32768_16]]
; CHECK:   %[[#S2B16_C15:]] = OpINotEqual %[[#BOOL]] %[[#S2B16_A15]] %[[#ZERO16]]
; CHECK:   %[[#S2B16_I15:]] = OpCompositeInsert %[[#BVEC16]] %[[#S2B16_C15]] %[[#S2B16_I14]] 15
;
; CHECK:   OpReturnValue %[[#S2B16_I15]]
define <16 x i1> @scalar_to_boolvec16(i16 %v) {
  %r = bitcast i16 %v to <16 x i1>
  ret <16 x i1> %r
}
