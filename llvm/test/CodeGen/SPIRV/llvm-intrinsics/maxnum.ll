; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s

define spir_func float @Test(float %x, float %y) {
entry:
  %0 = call float @llvm.maxnum.f32(float %x, float %y)
  ret float %0
}

; CHECK: OpFunction
; CHECK: %[[#x:]] = OpFunctionParameter %[[#]]
; CHECK: %[[#y:]] = OpFunctionParameter %[[#]]
; CHECK: %[[#res:]] = OpExtInst %[[#]] %[[#]] fmax %[[#x]] %[[#y]]
; CHECK: OpReturnValue %[[#res]]

declare float @llvm.maxnum.f32(float, float)
