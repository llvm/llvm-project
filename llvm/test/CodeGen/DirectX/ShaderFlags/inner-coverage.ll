; RUN: opt -S --passes="print-dx-shader-flags" 2>&1 %s | FileCheck %s
; RUN: llc %s --filetype=obj -o - | obj2yaml | FileCheck %s --check-prefix=DXC

target triple = "dxil-pc-shadermodel6.5-pixel"

; Reading SV_InnerCoverage sets the InnerCoverage flag (bit 10).

; CHECK: ; Combined Shader Flags for Module
; CHECK-NEXT: ; Shader Flags Value: 0x00000400
; CHECK-NEXT: ;
; CHECK-NEXT: ; Note: shader requires additional functionality:
; CHECK-NEXT: ;       PS Inner Coverage
; CHECK-NEXT: ; Note: extra DXIL module flags:
; CHECK-NEXT: ;
; CHECK-NEXT: ; Shader Flags for Module Functions

; CHECK: Function main : 0x00000400
define void @main() #0 {
  %ic = call i32 @llvm.dx.inner.coverage()
  ret void
}

attributes #0 = { "hlsl.shader"="pixel" }

; DXC: - Name:            SFI0
; DXC-NEXT:     Size:            8
; DXC-NEXT:     Flags:
; DXC:            InnerCoverage:   true
; DXC: ...
