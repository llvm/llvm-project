; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}
;
; Verify that bitcasts between bool vectors and other types are decomposed
; into element-wise operations instead of generating OpBitcast, which is
; invalid for OpTypeBool.

; CHECK-DAG: %[[#BOOL:]] = OpTypeBool
; CHECK-DAG: %[[#BVEC8:]] = OpTypeVector %[[#BOOL]] 8
; CHECK-DAG: %[[#I8:]] = OpTypeInt 8 0

; CHECK-DAG: %[[#ZERO:]] = OpConstantNull %[[#I8]]
; CHECK-DAG: %[[#ONE:]] = OpConstant %[[#I8]] 1
; CHECK-DAG: %[[#TWO:]] = OpConstant %[[#I8]] 2
; CHECK-DAG: %[[#FOUR:]] = OpConstant %[[#I8]] 4
; CHECK-DAG: %[[#EIGHT:]] = OpConstant %[[#I8]] 8
; CHECK-DAG: %[[#C16:]] = OpConstant %[[#I8]] 16
; CHECK-DAG: %[[#C32:]] = OpConstant %[[#I8]] 32
; CHECK-DAG: %[[#C64:]] = OpConstant %[[#I8]] 64
; CHECK-DAG: %[[#C128:]] = OpConstant %[[#I8]] 128
; CHECK-DAG: %[[#C3:]] = OpConstant %[[#I8]] 3
; CHECK-DAG: %[[#C5:]] = OpConstant %[[#I8]] 5
; CHECK-DAG: %[[#C6:]] = OpConstant %[[#I8]] 6
; CHECK-DAG: %[[#C7:]] = OpConstant %[[#I8]] 7

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
