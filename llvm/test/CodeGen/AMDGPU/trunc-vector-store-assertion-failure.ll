; RUN: llc < %s -march=r600 -mtriple=r600-- -mcpu=redwood | FileCheck %s

; This tests for a bug in the SelectionDAG where custom lowered truncated
; vector stores at the end of a basic block were not being added to the
; LegalizedNodes list, which triggered an assertion failure.

; CHECK-LABEL: {{^}}test:
; CHECK: MEM_RAT_CACHELESS STORE_RAW
define amdgpu_kernel void @test(ptr addrspace(1) %out, i32 %cond, <4 x i8> %in) {
entry:
  %0 = icmp eq i32 %cond, 0
  br i1 %0, label %if, label %done

if:
  store <4 x i8> %in, ptr addrspace(1) %out
  br label %done

done:
  ret void
}
