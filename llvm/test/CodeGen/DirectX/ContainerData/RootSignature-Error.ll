; RUN: not llc %s --filetype=obj -o - 2>&1 | FileCheck %s

target triple = "dxil-unknown-shadermodel6.0-compute"

; CHECK: LLVM ERROR: Invalid format for Root Signature Definition. Pairs of function, root signature expected.


define void @main() #0 {
entry:
  ret void
}

attributes #0 = { "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }


!dx.rootsignatures = !{!1} ; list of function/root signature pairs
!1= !{ !"RootFlags" } ; function, root signature
