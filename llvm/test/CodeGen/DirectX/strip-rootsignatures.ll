; RUN: opt -S -dxil-translate-metadata < %s | FileCheck %s

; Ensures that dxil-translate-metadata  will remove the dx.rootsignatures metadata

target triple = "dxil-unknown-shadermodel6.0-compute"

define void @main() {
entry:
  ret void
}

; CHECK-NOT: !dx.rootsignatures

!dx.rootsignatures = !{!2} ; list of function/root signature pairs
!2 = !{ ptr @main, !3, i32 2 } ; function, root signature
!3 = !{ !4 } ; list of root signature elements
!4 = !{ !"RootFlags", i32 1 } ; 1 = allow_input_assembler_input_layout
