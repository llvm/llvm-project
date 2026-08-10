; RUN: opt < %s -S -passes=globalopt,instcombine | FileCheck %s
;; check that global opt turns integers that only hold 0 or 1 into bools.

@G = internal addrspace(1) global i32 0
; CHECK: @G
; CHECK: addrspace(1)
; CHECK: global i1 false

define void @set1() {
  store i32 0, ptr addrspace(1) @G
; CHECK: store i1 false
  ret void
}

define void @set2() {
  store i32 1, ptr addrspace(1) @G
; CHECK: store i1 true
  ret void
}

define i1 @get() {
; CHECK-LABEL: @get(
  %A = load i32, ptr addrspace(1) @G
  %C = icmp slt i32 %A, 2
  ret i1 %C
; CHECK: ret i1 true
}

