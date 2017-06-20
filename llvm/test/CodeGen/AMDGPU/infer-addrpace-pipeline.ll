; RUN: opt -mtriple=amdgcn--amdhsa -disable-output -disable-verify -debug-pass=Structure -O2 %s 2>&1 | FileCheck -check-prefix=GCN %s

; GCN: Function Integration/Inlining
; GCN: FunctionPass Manager
; GCN: Infer address spaces
; GCN: SROA

define void @empty() {
  ret void
}
