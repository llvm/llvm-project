; RUN: llc -mtriple=x86_64-apple-darwin < %s| FileCheck %s

; A bug in DAGCombiner prevented it forming a zextload in this simple case
; because it counted both the chain user and the real user against the
; profitability total.

define void @load_zext(ptr nocapture %p){
entry:
  %0 = load i32, ptr %p, align 4
  %and = and i32 %0, 255
  tail call void @use(i32 %and)
  ret void
; CHECK: movzbl ({{%r[a-z]+}}), {{%e[a-z]+}}
}

declare void @use(i32)
