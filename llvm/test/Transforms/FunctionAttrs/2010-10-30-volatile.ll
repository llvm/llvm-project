; RUN: opt < %s -functionattrs -S | FileCheck %s
; PR8279

@g = constant i32 1

define void @foo() {
; CHECK: void @foo() #0 {
  %tmp = load volatile i32, i32* @g
  ret void
}

; CHECK: attributes #0 = { norecurse }
