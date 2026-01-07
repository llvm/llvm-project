; RUN: opt -S --passes="print-dx-shader-flags" 2>&1 %s | FileCheck %s
; RUN: llc %s --filetype=obj -o - | obj2yaml | FileCheck %s --check-prefix=DXC

; This test makes sure that resource arrays sizes are accounted for when
; counting the number of UAVs for setting the shader flag '64 UAV slots' when
; the validator version is >= 1.6

target triple = "dxil-pc-shadermodel6.7-library"

; CHECK:      Combined Shader Flags for Module
; CHECK-NEXT: Shader Flags Value: 0x00018000

; CHECK: Note: shader requires additional functionality:
; CHECK:        UAVs at every shader stage
; CHECK:        64 UAV slots

; CHECK: Function test : 0x00018000
define void @test() "hlsl.export" {
  ; RWBuffer<float> Buf : register(u0, space0)
  %buf0 = call target("dx.TypedBuffer", float, 1, 0, 1)
       @llvm.dx.resource.handlefrombinding.tdx.TypedBuffer_f32_1_0t(
           i32 0, i32 0, i32 1, i32 0, ptr null)

  ; RWBuffer<float> Buf[8] : register(u1, space0)
  %buf1 = call target("dx.TypedBuffer", float, 1, 0, 1)
       @llvm.dx.resource.handlefrombinding.tdx.TypedBuffer_f32_1_0t(
           i32 0, i32 1, i32 8, i32 0, ptr null)
  ret void
}

!dx.valver = !{!1}
!1 = !{i32 1, i32 6}

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"dx.resmayalias", i32 1}

; DXC: - Name:            SFI0
; DXC-NEXT:     Size:            8
; DXC-NEXT:     Flags:
; DXC:       Max64UAVs:         true
; DXC:       NextUnusedBit:   false
; DXC: ...
