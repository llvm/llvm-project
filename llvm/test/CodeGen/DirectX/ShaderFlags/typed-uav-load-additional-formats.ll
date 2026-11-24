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
      @llvm.dx.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, ptr null)
  %load = call {<4 x float>, i1} @llvm.dx.resource.load.typedbuffer(
      target("dx.TypedBuffer", <4 x float>, 1, 0, 0) %res, i32 0)
  %val = extractvalue {<4 x float>, i1} %load, 0
  ret <4 x float> %val
}

; CHECK: Function onecomponent : 0x00000000
define float @onecomponent() #0 {
  %res = call target("dx.TypedBuffer", float, 1, 0, 0)
      @llvm.dx.resource.handlefrombinding(i32 0, i32 1, i32 1, i32 0, ptr null)
  %load = call {float, i1} @llvm.dx.resource.load.typedbuffer(
      target("dx.TypedBuffer", float, 1, 0, 0) %res, i32 0)
  %val = extractvalue {float, i1} %load, 0
  ret float %val
}

; CHECK: Function texture2d_multicomponent : 0x00002000
define <4 x float> @texture2d_multicomponent(<2 x i32> %coords) #0 {
  %res = call target("dx.Texture", <4 x float>, 1, 0, 0, 2)
      @llvm.dx.resource.handlefrombinding(i32 0, i32 2, i32 1, i32 0, ptr null)
  %load = call <4 x float> @llvm.dx.resource.load.level(
      target("dx.Texture", <4 x float>, 1, 0, 0, 2) %res,
      <2 x i32> %coords, i32 poison, <2 x i32> zeroinitializer)
  ret <4 x float> %load
}

; CHECK: Function texture2darray_multicomponent : 0x00002000
define <4 x float> @texture2darray_multicomponent(<3 x i32> %coords) #0 {
  %res = call target("dx.Texture", <4 x float>, 1, 0, 0, 7)
      @llvm.dx.resource.handlefrombinding(i32 0, i32 3, i32 1, i32 0, ptr null)
  %load = call <4 x float> @llvm.dx.resource.load.level(
      target("dx.Texture", <4 x float>, 1, 0, 0, 7) %res,
      <3 x i32> %coords, i32 poison, <2 x i32> zeroinitializer)
  ret <4 x float> %load
}

; CHECK: Function texture_onecomponent : 0x00000000
define float @texture_onecomponent(<2 x i32> %coords) #0 {
  %res = call target("dx.Texture", float, 1, 0, 0, 2)
      @llvm.dx.resource.handlefrombinding(i32 0, i32 4, i32 1, i32 0, ptr null)
  %load = call float @llvm.dx.resource.load.level(
      target("dx.Texture", float, 1, 0, 0, 2) %res,
      <2 x i32> %coords, i32 poison, <2 x i32> zeroinitializer)
  ret float %load
}

; CHECK: Function noload : 0x00000000
define void @noload(<4 x float> %val) #0 {
  %res = call target("dx.TypedBuffer", <4 x float>, 1, 0, 0)
      @llvm.dx.resource.handlefrombinding(i32 0, i32 2, i32 1, i32 0, ptr null)
  call void @llvm.dx.resource.store.typedbuffer(
      target("dx.TypedBuffer", <4 x float>, 1, 0, 0) %res, i32 0,
      <4 x float> %val)
  ret void
}

!llvm.module.flags = !{!0}
!dx.valver = !{!1}
!0 = !{i32 1, !"dx.resmayalias", i32 1}
!1 = !{i32 1, i32 8}

attributes #0 = { convergent norecurse nounwind "hlsl.export"}

