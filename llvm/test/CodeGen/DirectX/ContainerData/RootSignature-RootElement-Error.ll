; RUN: not llc %s --filetype=obj -o - 2>&1 | FileCheck %s

target triple = "dxil-unknown-shadermodel6.0-compute"

; CHECK: LLVM ERROR: Missing Root Element Metadata Node.


define void @main() #0 {
entry:
  ret void
}

attributes #0 = { "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }


!dx.rootsignatures = !{!2} ; list of function/root signature pairs
!2 = !{ ptr @main, !3 } ; function, root signature
!3 = !{ !"NOTRootElements" } ; list of root signature elements
