; RUN: not opt -S \
; RUN:   -passes='spirv-emit-intrinsics,function(spirv-legalize-pointer-cast)' \
; RUN:   -mtriple=spirv-unknown-vulkan-compute %s -o - \
; RUN:   2>&1 | FileCheck %s

; CHECK: LLVM ERROR: Cannot lower vector loaded from scalar location with extract element usage of any but the first element.

@Values = external addrspace(3) global [4 x float]

define float @extractNonZero() {
  %vec = load <4 x float>, ptr addrspace(3) getelementptr inbounds
    (float, ptr addrspace(3) @Values, i64 1), align 4
  %elt = extractelement <4 x float> %vec, i32 2
  ret float %elt
}
