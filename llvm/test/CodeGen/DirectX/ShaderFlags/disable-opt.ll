; RUN: opt -S --passes="print-dx-shader-flags" 2>&1 %s | FileCheck %s

target triple = "dxilv1.6-unknown-shadermodel6.6-library"

; CHECK: ; Combined Shader Flags for Module
; CHECK-NEXT: ; Shader Flags Value: 0x00000001

; CHECK: ; Note: extra DXIL module flags:
; CHECK-NEXT: ;       D3D11_1_SB_GLOBAL_FLAG_SKIP_OPTIMIZATION

; CHECK: ; Shader Flags for Module Functions

target triple = "dxilv1.6-unknown-shadermodel6.6-library"

; CHECK: ; Function main : 0x00000000
define void @main() {
entry:
  ret void
}

!llvm.module.flags = !{!0}

!0 = !{i32 4, !"dx.disable_optimizations", i32 1}


