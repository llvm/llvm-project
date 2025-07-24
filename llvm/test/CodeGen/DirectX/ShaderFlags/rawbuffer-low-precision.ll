; RUN: opt -S --passes="print-dx-shader-flags" 2>&1 %s | FileCheck %s
; RUN: llc %s --filetype=obj -o - | obj2yaml | FileCheck %s --check-prefix=DXC

target triple = "dxil-pc-shadermodel6.7-library"

; CHECK: ; Combined Shader Flags for Module
; CHECK-NEXT: ; Shader Flags Value: 0x00800030
; CHECK-NEXT: ;
; CHECK-NEXT: ; Note: shader requires additional functionality:
; CHECK-NEXT: ;       Native low-precision data types
; CHECK-NEXT: ; Note: extra DXIL module flags:
; CHECK-NEXT: ;       Raw and structured buffers
; CHECK-NEXT: ;       Low-precision data types present
; CHECK-NEXT: ;       Enable native low-precision data types
; CHECK-NEXT: ;
; CHECK-NEXT: ; Shader Flags for Module Functions

; CHECK: Function rawbuf : 0x00800030
define void @rawbuf() "hlsl.export" {
  %halfrb = tail call target("dx.RawBuffer", <4 x half>, 0, 0)
    @llvm.dx.resource.handlefrombinding.tdx.RawBuffer_v4f16_0_0t(i32 0, i32 0, i32 1, i32 0, i1 false, ptr null)
  %i16rb = tail call target("dx.RawBuffer", <4 x i16>, 1, 0)
    @llvm.dx.resource.handlefrombinding.tdx.RawBuffer_v4i16_1_0t(i32 0, i32 1, i32 1, i32 0, i1 false, ptr null)
  %loadhalfrb = call { <4 x i16>, i1 }
    @llvm.dx.resource.load.rawbuffer.v4i16.tdx.RawBuffer_v4f16_0_0t(target("dx.RawBuffer", <4 x half>, 0, 0) %halfrb, i32 0, i32 0)
  %extracti16vec = extractvalue { <4 x i16>, i1 } %loadhalfrb, 0
  call void @llvm.dx.resource.store.rawbuffer.tdx.RawBuffer_v4i16_1_0t.v4i16(target("dx.RawBuffer", <4 x i16>, 1, 0) %i16rb, i32 0, i32 0, <4 x i16> %extracti16vec)
  ret void
}

; Metadata to avoid adding flags not currently of interest to this test, and
; enable native low precision data types
!dx.valver = !{!0}
!0 = !{i32 1, i32 8}
!llvm.module.flags = !{!1, !2}
!1 = !{i32 1, !"dx.nativelowprec", i32 1}
!2 = !{i32 1, !"dx.resmayalias", i32 1}

; DXC: - Name:            SFI0
; DXC-NEXT:     Size:            8
; DXC-NEXT:     Flags:
; DXC:      MinimumPrecision: false
; DXC:      NativeLowPrecision: true
; DXC: ...
