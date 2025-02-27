; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-SPIRV-DAG: %[[#TyInt32:]] = OpTypeInt 32 0
; CHECK-SPIRV-DAG: %[[#TyInt16:]] = OpTypeInt 16 0
; CHECK-SPIRV-DAG: %[[#TyHalf:]] = OpTypeFloat 16
; CHECK-SPIRV-DAG: %[[#Arg32:]] = OpFunctionParameter %[[#TyInt32]]
; CHECK-SPIRV-DAG: %[[#Arg16:]] = OpUConvert %[[#TyInt16]] %[[#Arg32]]
; CHECK-SPIRV-DAG: %[[#ValHalf:]] = OpBitcast %[[#TyHalf]] %[[#Arg16:]]
; CHECK-SPIRV-DAG: %[[#ValHalf2:]] = OpFMul %[[#TyHalf]] %[[#ValHalf]] %[[#ValHalf]]
; CHECK-SPIRV-DAG: %[[#Res16:]] = OpBitcast %[[#TyInt16]] %[[#ValHalf2]]
; CHECK-SPIRV-DAG: OpReturnValue %[[#Res16]]

define i16 @foo(i32 %arg) {
entry:
  %op16 = trunc i32 %arg to i16
  %val = bitcast i16 %op16 to half
  %val2 = fmul half %val, %val
  %res = bitcast half %val2 to i16
  ret i16 %res
}
