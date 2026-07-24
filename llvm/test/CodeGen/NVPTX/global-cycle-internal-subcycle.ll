; RUN: not --crash llc < %s -mtriple=nvptx64 -mcpu=sm_20 2>&1 | FileCheck %s

; A cyclic SCC is not necessarily made valid by the presence of a
; forward-declarable global. Declaring @outer breaks the larger loop, but the
; internal-only @inner_a <-> @inner_b subcycle still cannot be ordered.

; CHECK: LLVM ERROR: Circular dependency found in global variable set

@outer = addrspace(1) global ptr addrspace(1) @inner_a
@inner_a = internal addrspace(1) global { ptr addrspace(1), ptr addrspace(1) } {
  ptr addrspace(1) @outer,
  ptr addrspace(1) @inner_b
}
@inner_b = internal addrspace(1) global ptr addrspace(1) @inner_a
