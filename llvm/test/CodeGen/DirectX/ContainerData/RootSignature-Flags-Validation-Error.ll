; RUN: not llc %s --filetype=obj -o - 2>&1 | FileCheck %s

target triple = "dxil-unknown-shadermodel6.0-compute"

; CHECK: LLVM ERROR: Invalid flag value for RootFlag


define void @main() #0 {
entry:
  ret void
}

attributes #0 = { "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }


!dx.rootsignatures = !{!2} ; list of function/root signature pairs
!2 = !{ ptr @main, !3 } ; function, root signature
!3 = !{ !4 } ; list of root signature elements
!4 = !{ !"RootFlags", i32 2147487744 } ; 1 = allow_input_assembler_input_layout
