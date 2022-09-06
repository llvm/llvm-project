; RUN: llc %s -mtriple=spirv32-unknown-unknown -o - | FileCheck %s

declare float @llvm.fabs.f32(float)
declare float @llvm.rint.f32(float)
declare float @llvm.nearbyint.f32(float)
declare float @llvm.floor.f32(float)
declare float @llvm.ceil.f32(float)
declare float @llvm.round.f32(float)
declare float @llvm.trunc.f32(float)
declare float @llvm.sqrt.f32(float)
declare float @llvm.sin.f32(float)
declare float @llvm.cos.f32(float)
declare float @llvm.exp2.f32(float)
declare float @llvm.log.f32(float)
declare float @llvm.log10.f32(float)
declare float @llvm.log2.f32(float)
declare float @llvm.minnum.f32(float, float)
declare float @llvm.maxnum.f32(float, float)
declare <2 x half> @llvm.fabs.v2f16(<2 x half>)
declare <2 x half> @llvm.rint.v2f16(<2 x half>)
declare <2 x half> @llvm.nearbyint.v2f16(<2 x half>)
declare <2 x half> @llvm.floor.v2f16(<2 x half>)
declare <2 x half> @llvm.ceil.v2f16(<2 x half>)
declare <2 x half> @llvm.round.v2f16(<2 x half>)
declare <2 x half> @llvm.trunc.v2f16(<2 x half>)
declare <2 x half> @llvm.sqrt.v2f16(<2 x half>)
declare <2 x half> @llvm.sin.v2f16(<2 x half>)
declare <2 x half> @llvm.cos.v2f16(<2 x half>)
declare <2 x half> @llvm.exp2.v2f16(<2 x half>)
declare <2 x half> @llvm.log.v2f16(<2 x half>)
declare <2 x half> @llvm.log10.v2f16(<2 x half>)
declare <2 x half> @llvm.log2.v2f16(<2 x half>)

; CHECK-DAG: OpName %[[#SCALAR_FABS:]] "scalar_fabs"
; CHECK-DAG: OpName %[[#SCALAR_RINT:]] "scalar_rint"
; CHECK-DAG: OpName %[[#SCALAR_NEARBYINT:]] "scalar_nearbyint"
; CHECK-DAG: OpName %[[#SCALAR_FLOOR:]] "scalar_floor"
; CHECK-DAG: OpName %[[#SCALAR_CEIL:]] "scalar_ceil"
; CHECK-DAG: OpName %[[#SCALAR_ROUND:]] "scalar_round"
; CHECK-DAG: OpName %[[#SCALAR_TRUNC:]] "scalar_trunc"
; CHECK-DAG: OpName %[[#SCALAR_SQRT:]] "scalar_sqrt"
; CHECK-DAG: OpName %[[#SCALAR_SIN:]] "scalar_sin"
; CHECK-DAG: OpName %[[#SCALAR_COS:]] "scalar_cos"
; CHECK-DAG: OpName %[[#SCALAR_EXP2:]] "scalar_exp2"
; CHECK-DAG: OpName %[[#SCALAR_LOG:]] "scalar_log"
; CHECK-DAG: OpName %[[#SCALAR_LOG10:]] "scalar_log10"
; CHECK-DAG: OpName %[[#SCALAR_LOG2:]] "scalar_log2"
; CHECK-DAG: OpName %[[#SCALAR_MINNUM:]] "scalar_minnum"
; CHECK-DAG: OpName %[[#SCALAR_MAXNUM:]] "scalar_maxnum"
; CHECK-DAG: OpName %[[#VECTOR_FABS:]] "vector_fabs"
; CHECK-DAG: OpName %[[#VECTOR_RINT:]] "vector_rint"
; CHECK-DAG: OpName %[[#VECTOR_NEARBYINT:]] "vector_nearbyint"
; CHECK-DAG: OpName %[[#VECTOR_FLOOR:]] "vector_floor"
; CHECK-DAG: OpName %[[#VECTOR_CEIL:]] "vector_ceil"
; CHECK-DAG: OpName %[[#VECTOR_ROUND:]] "vector_round"
; CHECK-DAG: OpName %[[#VECTOR_TRUNC:]] "vector_trunc"
; CHECK-DAG: OpName %[[#VECTOR_SQRT:]] "vector_sqrt"
; CHECK-DAG: OpName %[[#VECTOR_SIN:]] "vector_sin"
; CHECK-DAG: OpName %[[#VECTOR_COS:]] "vector_cos"
; CHECK-DAG: OpName %[[#VECTOR_EXP2:]] "vector_exp2"
; CHECK-DAG: OpName %[[#VECTOR_LOG:]] "vector_log"
; CHECK-DAG: OpName %[[#VECTOR_LOG10:]] "vector_log10"
; CHECK-DAG: OpName %[[#VECTOR_LOG2:]] "vector_log2"

; CHECK-DAG: %[[#CLEXT:]] = OpExtInstImport "OpenCL.std"

; CHECK:      %[[#SCALAR_FABS]] = OpFunction
; CHECK-NEXT: %[[#A:]] = OpFunctionParameter
; CHECK:      OpLabel
; CHECK:      %[[#R:]] = OpExtInst %[[#]] %[[#CLEXT]] fabs %[[#A]]
; CHECK:      OpReturnValue %[[#R]]
; CHECK-NEXT: OpFunctionEnd
define float @scalar_fabs(float %a) {
    %r = call float @llvm.fabs.f32(float %a)
    ret float %r
}

; CHECK:      %[[#SCALAR_RINT]] = OpFunction
; CHECK-NEXT: %[[#A:]] = OpFunctionParameter
; CHECK:      OpLabel
; CHECK:      %[[#R:]] = OpExtInst %[[#]] %[[#CLEXT]] rint %[[#A]]
; CHECK:      OpReturnValue %[[#R]]
; CHECK-NEXT: OpFunctionEnd
define float @scalar_rint(float %a) {
    %r = call float @llvm.rint.f32(float %a)
    ret float %r
}

; CHECK:      %[[#SCALAR_NEARBYINT]] = OpFunction
; CHECK-NEXT: %[[#A:]] = OpFunctionParameter
; CHECK:      OpLabel
; CHECK:      %[[#R:]] = OpExtInst %[[#]] %[[#CLEXT]] rint %[[#A]]
; CHECK:      OpReturnValue %[[#R]]
; CHECK-NEXT: OpFunctionEnd
define float @scalar_nearbyint(float %a) {
    %r = call float @llvm.nearbyint.f32(float %a)
    ret float %r
}

; CHECK:      %[[#SCALAR_FLOOR]] = OpFunction
; CHECK-NEXT: %[[#A:]] = OpFunctionParameter
; CHECK:      OpLabel
; CHECK:      %[[#R:]] = OpExtInst %[[#]] %[[#CLEXT]] floor %[[#A]]
; CHECK:      OpReturnValue %[[#R]]
; CHECK-NEXT: OpFunctionEnd
define float @scalar_floor(float %a) {
    %r = call float @llvm.floor.f32(float %a)
    ret float %r
}

; CHECK:      %[[#SCALAR_CEIL]] = OpFunction
; CHECK-NEXT: %[[#A:]] = OpFunctionParameter
; CHECK:      OpLabel
; CHECK:      %[[#R:]] = OpExtInst %[[#]] %[[#CLEXT]] ceil %[[#A]]
; CHECK:      OpReturnValue %[[#R]]
; CHECK-NEXT: OpFunctionEnd
define float @scalar_ceil(float %a) {
    %r = call float @llvm.ceil.f32(float %a)
    ret float %r
}

; CHECK:      %[[#SCALAR_ROUND]] = OpFunction
; CHECK-NEXT: %[[#A:]] = OpFunctionParameter
; CHECK:      OpLabel
; CHECK:      %[[#R:]] = OpExtInst %[[#]] %[[#CLEXT]] round %[[#A]]
; CHECK:      OpReturnValue %[[#R]]
; CHECK-NEXT: OpFunctionEnd
define float @scalar_round(float %a) {
    %r = call float @llvm.round.f32(float %a)
    ret float %r
}

; CHECK:      %[[#SCALAR_TRUNC]] = OpFunction
; CHECK-NEXT: %[[#A:]] = OpFunctionParameter
; CHECK:      OpLabel
; CHECK:      %[[#R:]] = OpExtInst %[[#]] %[[#CLEXT]] trunc %[[#A]]
; CHECK:      OpReturnValue %[[#R]]
; CHECK-NEXT: OpFunctionEnd
define float @scalar_trunc(float %a) {
    %r = call float @llvm.trunc.f32(float %a)
    ret float %r
}

; CHECK:      %[[#SCALAR_SQRT]] = OpFunction
; CHECK-NEXT: %[[#A:]] = OpFunctionParameter
; CHECK:      OpLabel
; CHECK:      %[[#R:]] = OpExtInst %[[#]] %[[#CLEXT]] sqrt %[[#A]]
; CHECK:      OpReturnValue %[[#R]]
; CHECK-NEXT: OpFunctionEnd
define float @scalar_sqrt(float %a) {
    %r = call float @llvm.sqrt.f32(float %a)
    ret float %r
}

; CHECK:      %[[#SCALAR_SIN]] = OpFunction
; CHECK-NEXT: %[[#A:]] = OpFunctionParameter
; CHECK:      OpLabel
; CHECK:      %[[#R:]] = OpExtInst %[[#]] %[[#CLEXT]] sin %[[#A]]
; CHECK:      OpReturnValue %[[#R]]
; CHECK-NEXT: OpFunctionEnd
define float @scalar_sin(float %a) {
    %r = call float @llvm.sin.f32(float %a)
    ret float %r
}

; CHECK:      %[[#SCALAR_COS]] = OpFunction
; CHECK-NEXT: %[[#A:]] = OpFunctionParameter
; CHECK:      OpLabel
; CHECK:      %[[#R:]] = OpExtInst %[[#]] %[[#CLEXT]] cos %[[#A]]
; CHECK:      OpReturnValue %[[#R]]
; CHECK-NEXT: OpFunctionEnd
define float @scalar_cos(float %a) {
    %r = call float @llvm.cos.f32(float %a)
    ret float %r
}

; CHECK:      %[[#SCALAR_EXP2]] = OpFunction
; CHECK-NEXT: %[[#A:]] = OpFunctionParameter
; CHECK:      OpLabel
; CHECK:      %[[#R:]] = OpExtInst %[[#]] %[[#CLEXT]] exp2 %[[#A]]
; CHECK:      OpReturnValue %[[#R]]
; CHECK-NEXT: OpFunctionEnd
define float @scalar_exp2(float %a) {
    %r = call float @llvm.exp2.f32(float %a)
    ret float %r
}

; CHECK:      %[[#SCALAR_LOG]] = OpFunction
; CHECK-NEXT: %[[#A:]] = OpFunctionParameter
; CHECK:      OpLabel
; CHECK:      %[[#R:]] = OpExtInst %[[#]] %[[#CLEXT]] log %[[#A]]
; CHECK:      OpReturnValue %[[#R]]
; CHECK-NEXT: OpFunctionEnd
define float @scalar_log(float %a) {
    %r = call float @llvm.log.f32(float %a)
    ret float %r
}

; CHECK:      %[[#SCALAR_LOG10]] = OpFunction
; CHECK-NEXT: %[[#A:]] = OpFunctionParameter
; CHECK:      OpLabel
; CHECK:      %[[#R:]] = OpExtInst %[[#]] %[[#CLEXT]] log10 %[[#A]]
; CHECK:      OpReturnValue %[[#R]]
; CHECK-NEXT: OpFunctionEnd
define float @scalar_log10(float %a) {
    %r = call float @llvm.log10.f32(float %a)
    ret float %r
}

; CHECK:      %[[#SCALAR_LOG2]] = OpFunction
; CHECK-NEXT: %[[#A:]] = OpFunctionParameter
; CHECK:      OpLabel
; CHECK:      %[[#R:]] = OpExtInst %[[#]] %[[#CLEXT]] log2 %[[#A]]
; CHECK:      OpReturnValue %[[#R]]
; CHECK-NEXT: OpFunctionEnd
define float @scalar_log2(float %a) {
    %r = call float @llvm.log2.f32(float %a)
    ret float %r
}

; CHECK:      %[[#VECTOR_FABS]] = OpFunction
; CHECK-NEXT: %[[#A:]] = OpFunctionParameter
; CHECK:      OpLabel
; CHECK:      %[[#R:]] = OpExtInst %[[#]] %[[#CLEXT]] fabs %[[#A]]
; CHECK:      OpReturnValue %[[#R]]
; CHECK-NEXT: OpFunctionEnd
define <2 x half> @vector_fabs(<2 x half> %a) {
    %r = call <2 x half> @llvm.fabs.v2f16(<2 x half> %a)
    ret <2 x half> %r
}

; CHECK:      %[[#VECTOR_RINT]] = OpFunction
; CHECK-NEXT: %[[#A:]] = OpFunctionParameter
; CHECK:      OpLabel
; CHECK:      %[[#R:]] = OpExtInst %[[#]] %[[#CLEXT]] rint %[[#A]]
; CHECK:      OpReturnValue %[[#R]]
; CHECK-NEXT: OpFunctionEnd
define <2 x half> @vector_rint(<2 x half> %a) {
    %r = call <2 x half> @llvm.rint.v2f16(<2 x half> %a)
    ret <2 x half> %r
}

; CHECK:      %[[#VECTOR_NEARBYINT]] = OpFunction
; CHECK-NEXT: %[[#A:]] = OpFunctionParameter
; CHECK:      OpLabel
; CHECK:      %[[#R:]] = OpExtInst %[[#]] %[[#CLEXT]] rint %[[#A]]
; CHECK:      OpReturnValue %[[#R]]
; CHECK-NEXT: OpFunctionEnd
define <2 x half> @vector_nearbyint(<2 x half> %a) {
    %r = call <2 x half> @llvm.nearbyint.v2f16(<2 x half> %a)
    ret <2 x half> %r
}

; CHECK:      %[[#VECTOR_FLOOR]] = OpFunction
; CHECK-NEXT: %[[#A:]] = OpFunctionParameter
; CHECK:      OpLabel
; CHECK:      %[[#R:]] = OpExtInst %[[#]] %[[#CLEXT]] floor %[[#A]]
; CHECK:      OpReturnValue %[[#R]]
; CHECK-NEXT: OpFunctionEnd
define <2 x half> @vector_floor(<2 x half> %a) {
    %r = call <2 x half> @llvm.floor.v2f16(<2 x half> %a)
    ret <2 x half> %r
}

; CHECK:      %[[#VECTOR_CEIL]] = OpFunction
; CHECK-NEXT: %[[#A:]] = OpFunctionParameter
; CHECK:      OpLabel
; CHECK:      %[[#R:]] = OpExtInst %[[#]] %[[#CLEXT]] ceil %[[#A]]
; CHECK:      OpReturnValue %[[#R]]
; CHECK-NEXT: OpFunctionEnd
define <2 x half> @vector_ceil(<2 x half> %a) {
    %r = call <2 x half> @llvm.ceil.v2f16(<2 x half> %a)
    ret <2 x half> %r
}

; CHECK:      %[[#VECTOR_ROUND]] = OpFunction
; CHECK-NEXT: %[[#A:]] = OpFunctionParameter
; CHECK:      OpLabel
; CHECK:      %[[#R:]] = OpExtInst %[[#]] %[[#CLEXT]] round %[[#A]]
; CHECK:      OpReturnValue %[[#R]]
; CHECK-NEXT: OpFunctionEnd
define <2 x half> @vector_round(<2 x half> %a) {
    %r = call <2 x half> @llvm.round.v2f16(<2 x half> %a)
    ret <2 x half> %r
}

; CHECK:      %[[#VECTOR_TRUNC]] = OpFunction
; CHECK-NEXT: %[[#A:]] = OpFunctionParameter
; CHECK:      OpLabel
; CHECK:      %[[#R:]] = OpExtInst %[[#]] %[[#CLEXT]] trunc %[[#A]]
; CHECK:      OpReturnValue %[[#R]]
; CHECK-NEXT: OpFunctionEnd
define <2 x half> @vector_trunc(<2 x half> %a) {
    %r = call <2 x half> @llvm.trunc.v2f16(<2 x half> %a)
    ret <2 x half> %r
}

; CHECK:      %[[#VECTOR_SQRT]] = OpFunction
; CHECK-NEXT: %[[#A:]] = OpFunctionParameter
; CHECK:      OpLabel
; CHECK:      %[[#R:]] = OpExtInst %[[#]] %[[#CLEXT]] sqrt %[[#A]]
; CHECK:      OpReturnValue %[[#R]]
; CHECK-NEXT: OpFunctionEnd
define <2 x half> @vector_sqrt(<2 x half> %a) {
    %r = call <2 x half> @llvm.sqrt.v2f16(<2 x half> %a)
    ret <2 x half> %r
}

; CHECK:      %[[#VECTOR_SIN]] = OpFunction
; CHECK-NEXT: %[[#A:]] = OpFunctionParameter
; CHECK:      OpLabel
; CHECK:      %[[#R:]] = OpExtInst %[[#]] %[[#CLEXT]] sin %[[#A]]
; CHECK:      OpReturnValue %[[#R]]
; CHECK-NEXT: OpFunctionEnd
define <2 x half> @vector_sin(<2 x half> %a) {
    %r = call <2 x half> @llvm.sin.v2f16(<2 x half> %a)
    ret <2 x half> %r
}

; CHECK:      %[[#VECTOR_COS]] = OpFunction
; CHECK-NEXT: %[[#A:]] = OpFunctionParameter
; CHECK:      OpLabel
; CHECK:      %[[#R:]] = OpExtInst %[[#]] %[[#CLEXT]] cos %[[#A]]
; CHECK:      OpReturnValue %[[#R]]
; CHECK-NEXT: OpFunctionEnd
define <2 x half> @vector_cos(<2 x half> %a) {
    %r = call <2 x half> @llvm.cos.v2f16(<2 x half> %a)
    ret <2 x half> %r
}

; CHECK:      %[[#VECTOR_EXP2]] = OpFunction
; CHECK-NEXT: %[[#A:]] = OpFunctionParameter
; CHECK:      OpLabel
; CHECK:      %[[#R:]] = OpExtInst %[[#]] %[[#CLEXT]] exp2 %[[#A]]
; CHECK:      OpReturnValue %[[#R]]
; CHECK-NEXT: OpFunctionEnd
define <2 x half> @vector_exp2(<2 x half> %a) {
    %r = call <2 x half> @llvm.exp2.v2f16(<2 x half> %a)
    ret <2 x half> %r
}

; CHECK:      %[[#VECTOR_LOG]] = OpFunction
; CHECK-NEXT: %[[#A:]] = OpFunctionParameter
; CHECK:      OpLabel
; CHECK:      %[[#R:]] = OpExtInst %[[#]] %[[#CLEXT]] log %[[#A]]
; CHECK:      OpReturnValue %[[#R]]
; CHECK-NEXT: OpFunctionEnd
define <2 x half> @vector_log(<2 x half> %a) {
    %r = call <2 x half> @llvm.log.v2f16(<2 x half> %a)
    ret <2 x half> %r
}

; CHECK:      %[[#VECTOR_LOG10]] = OpFunction
; CHECK-NEXT: %[[#A:]] = OpFunctionParameter
; CHECK:      OpLabel
; CHECK:      %[[#R:]] = OpExtInst %[[#]] %[[#CLEXT]] log10 %[[#A]]
; CHECK:      OpReturnValue %[[#R]]
; CHECK-NEXT: OpFunctionEnd
define <2 x half> @vector_log10(<2 x half> %a) {
    %r = call <2 x half> @llvm.log10.v2f16(<2 x half> %a)
    ret <2 x half> %r
}

; CHECK:      %[[#VECTOR_LOG2]] = OpFunction
; CHECK-NEXT: %[[#A:]] = OpFunctionParameter
; CHECK:      OpLabel
; CHECK:      %[[#R:]] = OpExtInst %[[#]] %[[#CLEXT]] log2 %[[#A]]
; CHECK:      OpReturnValue %[[#R]]
; CHECK-NEXT: OpFunctionEnd
define <2 x half> @vector_log2(<2 x half> %a) {
    %r = call <2 x half> @llvm.log2.v2f16(<2 x half> %a)
    ret <2 x half> %r
}

; CHECK:      %[[#SCALAR_MINNUM]] = OpFunction
; CHECK-NEXT: %[[#A:]] = OpFunctionParameter
; CHECK-NEXT: %[[#B:]] = OpFunctionParameter
; CHECK:      OpLabel
; CHECK:      %[[#R:]] = OpExtInst %[[#]] %[[#CLEXT]] fmin %[[#A]] %[[#B]]
; CHECK:      OpReturnValue %[[#R]]
; CHECK-NEXT: OpFunctionEnd
define float @scalar_minnum(float %A, float %B) {
  %r = call float @llvm.minnum.f32(float %A, float %B)
  ret float %r
}

; CHECK:      %[[#SCALAR_MAXNUM]] = OpFunction
; CHECK-NEXT: %[[#A:]] = OpFunctionParameter
; CHECK-NEXT: %[[#B:]] = OpFunctionParameter
; CHECK:      OpLabel
; CHECK:      %[[#R:]] = OpExtInst %[[#]] %[[#CLEXT]] fmax %[[#A]] %[[#B]]
; CHECK:      OpReturnValue %[[#R]]
; CHECK-NEXT: OpFunctionEnd
define float @scalar_maxnum(float %A, float %B) {
  %r = call float @llvm.maxnum.f32(float %A, float %B)
  ret float %r
}
