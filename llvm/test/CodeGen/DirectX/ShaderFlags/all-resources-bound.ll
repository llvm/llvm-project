; RUN: opt -S --passes="print-dx-shader-flags" 2>&1 %s | FileCheck %s

; This test checks to ensure that setting the LLVM module flag
; "dx.allresourcesbound" to 1 sets the corresponding DXIL shader flag 

target triple = "dxil-pc-shadermodel6.8-library"

; CHECK:      Combined Shader Flags for Module
; CHECK-NEXT: Shader Flags Value: 0x00000100

; CHECK: Note: shader requires additional functionality:
; CHECK-NEXT: extra DXIL module flags:
; CHECK-NEXT:       All resources bound for the duration of shader execution


; CHECK: Function main : 0x00000100
define float @main() #0 {  
  ret float 0.0
}

!llvm.module.flags = !{!0}

!0 = !{i32 1, !"dx.allresourcesbound", i32 1}

attributes #0 = { convergent norecurse nounwind "hlsl.export"}
