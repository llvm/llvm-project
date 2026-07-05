; RUN: opt -mtriple=amdgcn-amd-amdhsa -S -passes=inline -inline-threshold=0 -debug-only=inline-cost %s -o - 2>&1 | FileCheck %s
; REQUIRES: asserts

; CHECK: Analyzing call of callee_not_only_one_live_use... (caller:caller)
; CHECK: Cost: -30
; CHECK: Analyzing call of callee_only_one_live_use... (caller:caller)
; CHECK: Cost: -165030

define internal void @callee_not_only_one_live_use() {
  ret void
}

define internal void @callee_only_one_live_use() {
  ret void
}

define void @caller() {
  call void @callee_not_only_one_live_use()
  call void @callee_not_only_one_live_use()
  call void @callee_only_one_live_use()
  ret void
}
