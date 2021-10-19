; RUN: opt < %s 2>&1 -disable-output \
; RUN: 	   -passes=inline -print-before-all -print-after-all | FileCheck %s
; RUN: opt < %s 2>&1 -disable-output \
; RUN: 	   -passes=inline -print-before-all -print-after-all -print-module-scope | FileCheck %s

; CHECK: IR Dump Before InlinerPass on (tester, foo)
; CHECK: IR Dump After InlinerPass on (tester, foo) (invalidated)
; CHECK: IR Dump Before InlinerPass on (tester)
; CHECK: IR Dump After InlinerPass on (tester)


define void @tester() noinline {
  call void @foo()
  ret void
}

define internal void @foo() alwaysinline {
  call void @tester()
  ret void
}
