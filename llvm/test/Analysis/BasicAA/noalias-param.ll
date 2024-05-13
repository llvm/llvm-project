; RUN: opt < %s -aa-pipeline=basic-aa -passes=aa-eval -print-all-alias-modref-info 2>&1 | FileCheck %s

declare ptr @captures(ptr %cap) nounwind readonly

; CHECK-LABEL: no
; CHECK: NoAlias:      i32* %a, i32* %b
define void @no(ptr noalias %a, ptr %b) nounwind {
entry:
  store i32 1, ptr %a
  %cap = call ptr @captures(ptr %a) nounwind readonly
  %l = load i32, ptr %b
  ret void
}

; CHECK-LABEL: yes
; CHECK: MayAlias:     i32* %c, i32* %d
define void @yes(ptr %c, ptr %d) nounwind {
entry:
  store i32 1, ptr %c 
  %cap = call ptr @captures(ptr %c) nounwind readonly
  %l = load i32, ptr %d
  ret void
}

; Result should be the same for byval instead of noalias.
; CHECK-LABEL: byval
; CHECK: NoAlias: i32* %a, i32* %b
define void @byval(ptr byval(i32) %a, ptr %b) nounwind {
entry:
  store i32 1, ptr %a
  %cap = call ptr @captures(ptr %a) nounwind readonly
  %l = load i32, ptr %b
  ret void
}
