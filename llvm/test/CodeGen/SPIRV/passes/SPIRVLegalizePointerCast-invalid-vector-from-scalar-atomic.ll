; RUN: not opt -S \
; RUN:   -passes='spirv-emit-intrinsics,function(spirv-legalize-pointer-cast)' \
; RUN:   -mtriple=spirv-unknown-vulkan-compute %s -o - \
; RUN:   2>&1 | FileCheck %s

; CHECK: LLVM ERROR: Cannot lower vector loaded from scalar with volatile or atomic load.

@Values = external addrspace(3) global [4 x float]

define <2 x float> @atomicLoad() {
  %value = load atomic <2 x float>, ptr addrspace(3) getelementptr inbounds
    (float, ptr addrspace(3) @Values, i64 1) monotonic, align 8
  ret <2 x float> %value
}
