; RUN: llc < %s -mcpu=sm_80 -mattr=+ptx73 -debug-only=isel -o /dev/null 2>&1 | FileCheck %s

; REQUIRES: asserts

target triple = "nvptx64-nvidia-cuda"

;; Selection DAG CSE is hard to test since we run CSE/GVN on the IR before and
;; after selection DAG ISel so most cases will be handled by one of these.
define void @foo(ptr %p) {
; CHECK-LABEL: Optimized legalized selection DAG: %bb.0 'foo:'
; CHECK:       addrspacecast[0 -> 5]
; CHECK-NOT:   addrspacecast[0 -> 5]
; CHECK-LABEL: ===== Instruction selection begins
;
  %a1 = addrspacecast ptr %p to ptr addrspace(5)
  call void @llvm.stackrestore(ptr %p)
  store ptr %p, ptr addrspace(5) %a1
  ret void
}
