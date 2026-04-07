; RUN: llc -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

define float @sitofp_f32_i32(i32 %in) {
; CHECK-DAG: OpName %[[#]] "conv"
; CHECK-DAG: OpConvertSToF %[[#]] %[[#]]
; CHECK-DAG: OpDecorate %[[#]] FPRoundingMode RTE
; CHECK-DAG: OpReturnValue %[[#]]
entry:
    %conv = tail call float @llvm.experimental.constrained.sitofp.f32.i32(i32 %in, metadata !"round.tonearest", metadata !"fpexcept.strict")
    ret float %conv
}

define float @sitofp_f32_i16(i16 %in) {
; CHECK-DAG: OpName %[[#]] "conv1"
; CHECK-DAG: OpConvertSToF %[[#]] %[[#]]
; CHECK-DAG: OpDecorate %[[#]] FPRoundingMode RTZ
; CHECK-DAG: OpReturnValue %[[#]]
entry:
    %conv1 = tail call float @llvm.experimental.constrained.sitofp.f32.i16(i16 %in, metadata !"round.towardzero", metadata !"fpexcept.strict")
    ret float %conv1
}

define double @sitofp_f64_i16(i16 %in) {
; CHECK-DAG: OpName %[[#]] "conv2"
; CHECK-DAG: OpConvertSToF %[[#]] %[[#]]
; CHECK-DAG: OpDecorate %[[#]] FPRoundingMode RTP
; CHECK-DAG: OpReturnValue %[[#]]
entry:
    %conv2 = tail call double @llvm.experimental.constrained.sitofp.f64.i16(i16 %in, metadata !"round.upward", metadata !"fpexcept.strict")
    ret double %conv2
}

define double @sitofp_f64_i32(i16 %in) {
; CHECK-DAG: OpName %[[#]] "conv3"
; CHECK-DAG: OpConvertSToF %[[#]] %[[#]]
; CHECK-DAG: OpDecorate %[[#]] FPRoundingMode RTN
; CHECK-DAG: OpReturnValue %[[#]]
entry:
    %conv3 = tail call double @llvm.experimental.constrained.sitofp.f64.i32(i16 %in, metadata !"round.downward", metadata !"fpexcept.strict")
    ret double %conv3
}

define <4 x double> @sitofp_v4f64_v4i32(<4 x i16> %in) {
; CHECK-DAG: OpName %[[#]] "conv4"
; CHECK-DAG: OpConvertSToF %[[#]] %[[#]]
; CHECK-DAG: OpDecorate %[[#]] FPRoundingMode RTE
; CHECK-DAG: OpReturnValue %[[#]]
entry:
    %conv4 = tail call <4 x double> @llvm.experimental.constrained.sitofp.v4f64.v4i32(<4 x i16> %in, metadata !"round.tonearest", metadata !"fpexcept.strict")
    ret <4 x double> %conv4
}

define float @uitofp_f32_i32(i32 %in) {
; CHECK-DAG: OpName %[[#]] "conv5"
; CHECK-DAG: OpConvertUToF %[[#]] %[[#]]
; CHECK-DAG: OpDecorate %[[#]] FPRoundingMode RTE
; CHECK-DAG: OpReturnValue %[[#]]
entry:
    %conv5 = tail call float @llvm.experimental.constrained.uitofp.f32.i32(i32 %in, metadata !"round.tonearest", metadata !"fpexcept.strict")
    ret float %conv5
}

define double @uitofp_f64_i32(i32 %in) {
; CHECK-DAG: OpName %[[#]] "conv6"
; CHECK-DAG: OpConvertUToF %[[#]] %[[#]]
; CHECK-DAG: OpDecorate %[[#]] FPRoundingMode RTZ
; CHECK-DAG: OpReturnValue %[[#]]
entry:
    %conv6 = tail call double @llvm.experimental.constrained.uitofp.f64.i32(i32 %in, metadata !"round.towardzero", metadata !"fpexcept.strict")
    ret double %conv6
}

define float @uitofp_f32_i16(i16 %in) {
; CHECK-DAG: OpName %[[#]] "conv7"
; CHECK-DAG: OpConvertUToF %[[#]] %[[#]]
; CHECK-DAG: OpDecorate %[[#]] FPRoundingMode RTP
; CHECK-DAG: OpReturnValue %[[#]]
entry:
    %conv7 = tail call float @llvm.experimental.constrained.uitofp.f32.i16(i16 %in, metadata !"round.upward", metadata !"fpexcept.strict")
    ret float %conv7
}

define double @uitofp_f64_i16(i16 %in) {
; CHECK-DAG: OpName %[[#]] "conv8"
; CHECK-DAG: OpConvertUToF %[[#]] %[[#]]
; CHECK-DAG: OpDecorate %[[#]] FPRoundingMode RTN
; CHECK-DAG: OpReturnValue %[[#]]
entry:
    %conv8 = tail call double @llvm.experimental.constrained.uitofp.f64.i16(i16 %in, metadata !"round.downward", metadata !"fpexcept.strict")
    ret double %conv8
}

define <4 x float> @uitofp_v4f32_v4i32(<4 x i32> %in) {
; CHECK-DAG: OpName %[[#]] "conv9"
; CHECK-DAG: OpConvertUToF %[[#]] %[[#]]
; CHECK-DAG: OpDecorate %[[#]] FPRoundingMode RTE
; CHECK-DAG: OpReturnValue %[[#]]
entry:
    %conv9 = tail call <4 x float> @llvm.experimental.constrained.uitofp.v4f32.v4i32(<4 x i32> %in, metadata !"round.tonearest", metadata !"fpexcept.strict")
    ret <4 x float> %conv9
}

define i32 @fptosi_i32_f32(float %in) {
; CHECK-DAG: OpName %[[#]] "conv10"
; CHECK-DAG: OpConvertFToS %[[#]] %[[#]]
; CHECK-DAG: OpReturnValue %[[#]]
entry:
    %conv10 = tail call i32 @llvm.experimental.constrained.fptosi.i32.f32(float %in, metadata !"fpexcept.strict")
    ret i32 %conv10
}

define i32 @fptosi_i32_f64(double %in) {
; CHECK-DAG: OpName %[[#]] "conv11"
; CHECK-DAG: OpConvertFToS %[[#]] %[[#]]
; CHECK-DAG: OpReturnValue %[[#]]
entry:
    %conv11 = tail call i32 @llvm.experimental.constrained.fptosi.i32.f64(double %in, metadata !"fpexcept.strict")
    ret i32 %conv11
}

define i16 @fptosi_i16_f32(float %in) {
; CHECK-DAG: OpName %[[#]] "conv12"
; CHECK-DAG: OpConvertFToS %[[#]] %[[#]]
; CHECK-DAG: OpReturnValue %[[#]]
entry:
    %conv12 = tail call i16 @llvm.experimental.constrained.fptosi.i16.f64(float %in, metadata !"fpexcept.strict")
    ret i16 %conv12
}

define i16 @fptosi_i16_f64(double %in) {
; CHECK-DAG: OpName %[[#]] "conv13"
; CHECK-DAG: OpConvertFToS %[[#]] %[[#]]
; CHECK-DAG: OpReturnValue %[[#]]
entry:
    %conv13 = tail call i16 @llvm.experimental.constrained.fptosi.i16.f64(double %in, metadata !"fpexcept.strict")
    ret i16 %conv13
}

define <4 x i16> @fptosi_v4i16_v4f64(<4 x double> %in) {
; CHECK-DAG: OpName %[[#]] "conv14"
; CHECK-DAG: OpConvertFToS %[[#]] %[[#]]
; CHECK-DAG: OpReturnValue %[[#]]
entry:
    %conv14 = tail call <4 x i16> @llvm.experimental.constrained.fptosi.v4i16.v4f64(<4 x double> %in, metadata !"fpexcept.strict")
    ret <4 x i16> %conv14
}

define i32 @fptoui_i32_f32(float %in) {
; CHECK-DAG: OpName %[[#]] "conv15"
; CHECK-DAG: OpConvertFToU %[[#]] %[[#]]
; CHECK-DAG: OpReturnValue %[[#]]
entry:
    %conv15 = tail call i32 @llvm.experimental.constrained.fptoui.i32.f32(float %in, metadata !"fpexcept.strict")
    ret i32 %conv15
}

define i32 @fptoui_i32_f64(double %in) {
; CHECK-DAG: OpName %[[#]] "conv16"
; CHECK-DAG: OpConvertFToU %[[#]] %[[#]]
; CHECK-DAG: OpReturnValue %[[#]]
entry:
    %conv16 = tail call i32 @llvm.experimental.constrained.fptoui.i32.f64(double %in, metadata !"fpexcept.strict")
    ret i32 %conv16
}

define i16 @fptoui_i16_f32(float %in) {
; CHECK-DAG: OpName %[[#]] "conv17"
; CHECK-DAG: OpConvertFToU %[[#]] %[[#]]
; CHECK-DAG: OpReturnValue %[[#]]
entry:
    %conv17 = tail call i16 @llvm.experimental.constrained.fptoui.i16.f32(float %in, metadata !"fpexcept.strict")
    ret i16 %conv17
}

define i16 @fptoui_i16_f64(double %in) {
; CHECK-DAG: OpName %[[#]] "conv18"
; CHECK-DAG: OpConvertFToU %[[#]] %[[#]]
; CHECK-DAG: OpReturnValue %[[#]]
entry:
    %conv18 = tail call i16 @llvm.experimental.constrained.fptoui.i16.f64(double %in, metadata !"fpexcept.strict")
    ret i16 %conv18
}

define <4 x i32> @fptoui_v4i32_v4f64(<4 x double> %in) {
; CHECK-DAG: OpName %[[#]] "conv19"
; CHECK-DAG: OpConvertFToU %[[#]] %[[#]]
; CHECK-DAG: OpReturnValue %[[#]]
entry:
    %conv19 = tail call <4 x i32> @llvm.experimental.constrained.fptoui.v4i32.v4f64(<4 x double> %in, metadata !"fpexcept.strict")
    ret <4 x i32> %conv19
}

define double @fpext_f64_f32(float %in) {
; CHECK-DAG: OpName %[[#]] "conv20"
; CHECK-DAG: OpFConvert %[[#]] %[[#]]
; CHECK-DAG: OpReturnValue %[[#]]
entry:
    %conv20 = tail call double @llvm.experimental.constrained.fpext.f64.f32(float %in, metadata !"fpexcept.strict")
    ret double %conv20
}

define <4 x double> @fpext_v4f64_v4f32(<4 x float> %in) {
; CHECK-DAG: OpName %[[#]] "conv21"
; CHECK-DAG: OpFConvert %[[#]] %[[#]]
; CHECK-DAG: OpReturnValue %[[#]]
entry:
    %conv21 = tail call <4 x double> @llvm.experimental.constrained.fpext.v4f64.v4f32(<4 x float> %in, metadata !"fpexcept.strict")
    ret <4 x double> %conv21
}

define <4 x float> @fptrunc_v4f32_v4f64(<4 x double> %in) {
; CHECK-DAG: OpName %[[#]] "conv22"
; CHECK-DAG: OpFConvert %[[#]] %[[#]]
; CHECK-DAG: OpDecorate %[[#]] FPRoundingMode RTZ
; CHECK-DAG: OpReturnValue %[[#]]
entry:
    %conv22 = tail call <4 x float> @llvm.experimental.constrained.fptrunc.v4f32.v4f64(<4 x double> %in, metadata !"round.towardzero", metadata !"fpexcept.strict")
    ret <4 x float> %conv22
}

define float @fptrunc_f32_f64(double %in) {
; CHECK-DAG: OpName %[[#]] "conv23"
; CHECK-DAG: OpFConvert %[[#]] %[[#]]
; CHECK-DAG: OpDecorate %[[#]] FPRoundingMode RTE
; CHECK-DAG: OpReturnValue %[[#]]
entry:
    %conv23 = tail call float @llvm.experimental.constrained.fptrunc.f32.f64(double %in, metadata !"round.tonearest", metadata !"fpexcept.strict")
    ret float %conv23
}
