; RUN: llc < %s -O0 -debug-only=isel -o /dev/null 2>&1 | FileCheck %s

; REQUIRES: asserts

target triple = "nvptx64-nvidia-cuda"

;; Selection DAG CSE is hard to test since we run CSE/GVN on the IR before and
;; after selection DAG ISel so most cases will be handled by one of these.
define void @foo(ptr %p) {
; CHECK-LABEL: Initial selection DAG
;
; CHECK:  [[ASC:t[0-9]+]]{{.*}} = addrspacecast
; CHECK:                          store{{.*}} [[ASC]]
; CHECK:                          store{{.*}} [[ASC]]
;
; CHECK-LABEL: Optimized lowered selection
;
   %a1 = addrspacecast ptr %p to ptr addrspace(5)
   %a2 = addrspacecast ptr %p to ptr addrspace(5)
   store i32 0, ptr addrspace(5) %a1
   store i32 0, ptr addrspace(5) %a2
   ret void
}
