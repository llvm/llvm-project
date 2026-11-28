; RUN: not opt -S \
; RUN:   -passes='spirv-emit-intrinsics,function(spirv-legalize-pointer-cast)' \
; RUN:   -mtriple=spirv-unknown-vulkan-compute %s -o - \
; RUN:   2>&1 | FileCheck %s

; CHECK: LLVM ERROR: Cannot lower vector loaded from scalar with volatile or atomic load.

@Values = external addrspace(3) global [4 x float]

define <4 x float> @volatileLoad() {
  %value = load volatile <4 x float>, ptr addrspace(3) getelementptr inbounds
    (float, ptr addrspace(3) @Values, i64 1), align 4
  ret <4 x float> %value
}
