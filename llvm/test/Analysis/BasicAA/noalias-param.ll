; RUN: opt < %s -aa-pipeline=basic-aa -passes=aa-eval -print-all-alias-modref-info 2>&1 | FileCheck %s

declare i32* @captures(i32* %cap) nounwind readonly

; CHECK-LABEL: no
; CHECK: NoAlias:      i32* %a, i32* %b
define void @no(i32* noalias %a, i32* %b) nounwind {
entry:
  store i32 1, i32* %a
  %cap = call i32* @captures(i32* %a) nounwind readonly
  %l = load i32, i32* %b
  ret void
}

; CHECK-LABEL: yes
; CHECK: MayAlias:     i32* %c, i32* %d
define void @yes(i32* %c, i32* %d) nounwind {
entry:
  store i32 1, i32* %c 
  %cap = call i32* @captures(i32* %c) nounwind readonly
  %l = load i32, i32* %d
  ret void
}

; Result should be the same for byval instead of noalias.
; CHECK-LABEL: byval
; CHECK: NoAlias: i32* %a, i32* %b
define void @byval(i32* byval(i32) %a, i32* %b) nounwind {
entry:
  store i32 1, i32* %a
  %cap = call i32* @captures(i32* %a) nounwind readonly
  %l = load i32, i32* %b
  ret void
}
