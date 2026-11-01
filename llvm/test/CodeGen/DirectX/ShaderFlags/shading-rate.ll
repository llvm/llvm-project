; RUN: split-file %s %t
; RUN: opt -S --passes="print-dx-shader-flags" 2>&1 %t/vs-out.ll | FileCheck %s --check-prefix=VS-OUT
; RUN: opt -S --passes="print-dx-shader-flags" 2>&1 %t/ps-in.ll | FileCheck %s --check-prefix=PS-IN
; RUN: opt -S --passes="print-dx-shader-flags" 2>&1 %t/none.ll | FileCheck %s --check-prefix=NONE

; Declaring SV_ShadingRate in either signature sets the ShadingRate shader
; feature flag (bit 24), even when the shader body never touches it.

;--- vs-out.ll
target triple = "dxil-pc-shadermodel6.8-vertex"

; VS-OUT: ; Combined Shader Flags for Module
; VS-OUT-NEXT: ; Shader Flags Value: 0x01000000
; VS-OUT-NEXT: ;
; VS-OUT-NEXT: ; Note: shader requires additional functionality:
; VS-OUT-NEXT: ;       Shading Rate
; VS-OUT-NEXT: ; Note: extra DXIL module flags:
; VS-OUT-NEXT: ;
; VS-OUT-NEXT: ; Shader Flags for Module Functions

; VS-OUT: Function main : 0x01000000
define void @main() #0 {
  ret void
}

attributes #0 = { "hlsl.shader"="vertex" }

!dx.valver = !{!20}
!20 = !{i32 1, i32 8}
!dx.semantic.signatures = !{!0}
!0 = !{ptr @main, null, !1}
!1 = !{!2}
!2 = !{i32 0, !"SV_SHADINGRATE", i32 5, i32 29, !3, i32 0, i32 1, i8 1, i32 0, i8 0, i8 0, i8 0, i32 0}
!3 = !{i32 0}

;--- ps-in.ll
target triple = "dxil-pc-shadermodel6.8-pixel"

; PS-IN: ; Combined Shader Flags for Module
; PS-IN-NEXT: ; Shader Flags Value: 0x01000000
; PS-IN-NEXT: ;
; PS-IN-NEXT: ; Note: shader requires additional functionality:
; PS-IN-NEXT: ;       Shading Rate
; PS-IN-NEXT: ; Note: extra DXIL module flags:
; PS-IN-NEXT: ;
; PS-IN-NEXT: ; Shader Flags for Module Functions

; PS-IN: Function main : 0x01000000
define void @main() #0 {
  ret void
}

attributes #0 = { "hlsl.shader"="pixel" }

!dx.valver = !{!20}
!20 = !{i32 1, i32 8}
!dx.semantic.signatures = !{!0}
!0 = !{ptr @main, !1, null}
!1 = !{!2}
; Integer pixel inputs require constant interpolation (mode 1).
!2 = !{i32 0, !"SV_SHADINGRATE", i32 5, i32 29, !3, i32 1, i32 1, i8 1, i32 0, i8 0, i8 0, i8 0, i32 0}
!3 = !{i32 0}

;--- none.ll
target triple = "dxil-pc-shadermodel6.8-vertex"

; A signature without SV_ShadingRate must not set the flag.
; NONE: ; Combined Shader Flags for Module
; NONE-NEXT: ; Shader Flags Value: 0x00000000
; NONE-NOT: Shading Rate
; NONE: Function main : 0x00000000
define void @main() #0 {
  ret void
}

attributes #0 = { "hlsl.shader"="vertex" }

!dx.valver = !{!20}
!20 = !{i32 1, i32 8}
!dx.semantic.signatures = !{!0}
!0 = !{ptr @main, null, !1}
!1 = !{!2}
!2 = !{i32 0, !"SV_POSITION", i32 9, i32 3, !3, i32 0, i32 1, i8 4, i32 0, i8 0, i8 0, i8 0, i32 0}
!3 = !{i32 0}
