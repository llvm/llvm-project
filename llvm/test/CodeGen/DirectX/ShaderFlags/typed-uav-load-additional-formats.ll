; RUN: opt -S --passes="print-dx-shader-flags" 2>&1 %s | FileCheck %s
; RUN: llc %s --filetype=obj -o - | obj2yaml | FileCheck %s --check-prefix=CHECK-OBJ

target triple = "dxil-pc-shadermodel6.7-library"

; CHECK-OBJ: - Name: SFI0
; CHECK-OBJ:   Flags:
; CHECK-OBJ:     TypedUAVLoadAdditionalFormats: true

; CHECK:      Combined Shader Flags for Module
; CHECK-NEXT: Shader Flags Value: 0x00002000

; CHECK: Note: shader requires additional functionality:
; CHECK:       Typed UAV Load Additional Formats

; CHECK: Function multicomponent : 0x00002000
define <4 x float> @multicomponent() #0 {
  %res = call target("dx.TypedBuffer", <4 x float>, 1, 0, 0)
      @llvm.dx.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, i1 false)
  %load = call {<4 x float>, i1} @llvm.dx.resource.load.typedbuffer(
      target("dx.TypedBuffer", <4 x float>, 1, 0, 0) %res, i32 0)
  %val = extractvalue {<4 x float>, i1} %load, 0
  ret <4 x float> %val
}

; CHECK: Function onecomponent : 0x00000000
define float @onecomponent() #0 {
  %res = call target("dx.TypedBuffer", float, 1, 0, 0)
      @llvm.dx.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, i1 false)
  %load = call {float, i1} @llvm.dx.resource.load.typedbuffer(
      target("dx.TypedBuffer", float, 1, 0, 0) %res, i32 0)
  %val = extractvalue {float, i1} %load, 0
  ret float %val
}

; CHECK: Function noload : 0x00000000
define void @noload(<4 x float> %val) #0 {
  %res = call target("dx.TypedBuffer", <4 x float>, 1, 0, 0)
      @llvm.dx.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, i1 false)
  call void @llvm.dx.resource.store.typedbuffer(
      target("dx.TypedBuffer", <4 x float>, 1, 0, 0) %res, i32 0,
      <4 x float> %val)
  ret void
}

attributes #0 = { convergent norecurse nounwind "hlsl.export"}
