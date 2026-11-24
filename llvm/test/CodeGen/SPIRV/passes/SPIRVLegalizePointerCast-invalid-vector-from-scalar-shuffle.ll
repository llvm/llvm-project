; RUN: not opt -S \
; RUN:   -passes='spirv-emit-intrinsics,function(spirv-legalize-pointer-cast)' \
; RUN:   -mtriple=spirv-unknown-vulkan-compute %s -o - \
; RUN:   2>&1 | FileCheck %s

; CHECK: LLVM ERROR: Cannot lower vector loaded from scalar location with shufflevector usage of any but index 0 or poison.

@Values = external addrspace(3) global [4 x float]

define <4 x float> @shuffleNonZero() {
  %vec = load <4 x float>, ptr addrspace(3) getelementptr inbounds
    (float, ptr addrspace(3) @Values, i64 1), align 4
  %shuf = shufflevector <4 x float> %vec, <4 x float> poison,
                        <4 x i32> <i32 0, i32 poison, i32 1, i32 poison>
  ret <4 x float> %shuf
}
