; RUN: split-file %s %t
; RUN: opt -S --dxil-translate-metadata %t/present.ll 2>&1 | FileCheck %t/present.ll
; RUN: opt -S --dxil-translate-metadata %t/missing.ll 2>&1 | FileCheck %t/missing.ll

; Test that !llvm.ident is preserved when a frontend emits it, and that
; a warning is produced when it is absent.

;--- present.ll

; CHECK-NOT: missing !llvm.ident
; CHECK-DAG: !llvm.ident = !{![[#IDENT:]]}
; CHECK-DAG: ![[#IDENT]] = !{!"frontend v1.0"}

target triple = "dxil-pc-shadermodel6.6-compute"

define void @CSMain() #0 {
  ret void
}

attributes #0 = { "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }

!llvm.ident = !{!0}
!0 = !{!"frontend v1.0"}

;--- missing.ll

; CHECK: warning: {{.*}}missing !llvm.ident metadata; frontend should emit it

target triple = "dxil-pc-shadermodel6.6-compute"

define void @CSMain() #0 {
  ret void
}

attributes #0 = { "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }
