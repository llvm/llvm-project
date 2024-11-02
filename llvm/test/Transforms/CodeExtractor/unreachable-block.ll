; RUN: opt -S -passes=partial-inliner %s | FileCheck %s

; CHECK-LABEL: define void @dipsy(
; CHECK-NEXT:   call void @tinkywinky.1.ontrue()
; CHECK-NEXT:   call void @patatuccio()
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK-LABEL: define internal void @tinkywinky.1.ontrue() {
; CHECK-NEXT: newFuncRoot:
; CHECK-NEXT:   br label %ontrue
; CHECK: ontrue:
; CHECK-NEXT:   call void @patatino()
; CHECK-NEXT:   br label %onfalse{{.*}}
; CHECK: onfalse{{.*}}:
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

declare void @patatino()
declare void @patatuccio()

define fastcc void @tinkywinky() {
  br i1 true, label %ontrue, label %onfalse
ontrue:
  call void @patatino()
  br label %onfalse
onfalse:
  call void @patatuccio()
  ret void
cantreachme:
  ret void
}
define void @dipsy() {
  call fastcc void @tinkywinky()
  ret void
}
