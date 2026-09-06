; RUN: not --crash llc -filetype=null -mtriple=x86_64 -O0 -print-changed=inst-quiet %s 2>&1 | FileCheck %s
; RUN: not --crash llc -enable-new-pm=1 -filetype=null -mtriple=x86_64 -O0 -print-changed=inst-quiet %s 2>&1 | FileCheck %s

define void @f() {
  ret void
}

; CHECK: LLVM ERROR: instruction-level change printing is only supported for LLVM IR under the new pass manager
