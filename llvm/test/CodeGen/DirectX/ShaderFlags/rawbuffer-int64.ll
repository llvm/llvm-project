; RUN: opt -S --passes="print-dx-shader-flags" 2>&1 %s | FileCheck %s
; RUN: llc %s --filetype=obj -o - | obj2yaml | FileCheck %s --check-prefix=DXC

target triple = "dxil-pc-shadermodel6.7-library"

; CHECK: ; Combined Shader Flags for Module
; CHECK-NEXT: ; Shader Flags Value: 0x00100010
; CHECK-NEXT: ;
; CHECK-NEXT: ; Note: shader requires additional functionality:
; CHECK-NEXT: ;       64-Bit integer
; CHECK-NEXT: ; Note: extra DXIL module flags:
; CHECK-NEXT: ;       Raw and structured buffers
; CHECK-NEXT: ;
; CHECK-NEXT: ; Shader Flags for Module Functions

; CHECK: Function rawbuf : 0x00100010
define void @rawbuf() "hlsl.export" {
  %rb = tail call target("dx.RawBuffer", <4 x i64>, 0, 0)
    @llvm.dx.resource.handlefrombinding.tdx.RawBuffer_v4f16_0_0t(i32 0, i32 0, i32 1, i32 0, ptr null)
  %load = call { <4 x i64>, i1 }
    @llvm.dx.resource.load.rawbuffer.v4i64.tdx.RawBuffer_v4f16_0_0t(target("dx.RawBuffer", <4 x i64>, 0, 0) %rb, i32 0, i32 0)
  %extract = extractvalue { <4 x i64>, i1 } %load, 0
  ret void
}

; Metadata to avoid adding flags not currently of interest to this test
!dx.valver = !{!0}
!0 = !{i32 1, i32 8}
!llvm.module.flags = !{!1}
!1 = !{i32 1, !"dx.resmayalias", i32 1}

; DXC: - Name:            SFI0
; DXC-NEXT:     Size:            8
; DXC-NEXT:     Flags:
; DXC:            Int64Ops:        true
; DXC: ...
