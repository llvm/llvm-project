; RUN: opt -S --passes="print-dx-shader-flags" 2>&1 %s | FileCheck %s
; RUN: llc %s --filetype=obj -o - | obj2yaml | FileCheck %s --check-prefix=OBJ

; This test makes sure that Max64UAVs is correctly set when using an
; unbounded array of UAVs;

target triple = "dxil-pc-shadermodel6.7-library"

; CHECK:      Combined Shader Flags for Module
; CHECK-NEXT: Shader Flags Value: 0x200018000

; CHECK: Note: shader requires additional functionality:
; CHECK:        UAVs at every shader stage
; CHECK:        64 UAV slots

; CHECK: Function test : 0x200018000
define void @test() "hlsl.export" {
  ; RWBuffer<float> Buf[] : register(u1, space0)
  %buf1 = call target("dx.TypedBuffer", float, 1, 0, 1)
       @llvm.dx.resource.handlefrombinding.tdx.TypedBuffer_f32_1_0t(
           i32 0, i32 1, i32 0, i32 0, ptr null)
  ret void
}

!dx.valver = !{!1}
!1 = !{i32 1, i32 6}

; OBJ:      - Name: SFI0
; OBJ-NEXT:     Size: 8
; OBJ-NEXT:     Flags:
; OBJ:       Max64UAVs: true
