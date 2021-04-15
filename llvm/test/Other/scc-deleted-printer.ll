; RUN: opt < %s 2>&1 -disable-output \
; RUN: 	   -passes=inline -print-before-all -print-after-all | FileCheck %s -check-prefix=INL
; RUN: opt < %s 2>&1 -disable-output \
; RUN: 	   -passes=inline -print-before-all -print-after-all -print-module-scope | FileCheck %s -check-prefix=INL-MOD

; INL: IR Dump Before InlinerPass on (tester, foo)
; INL-NOT: IR Dump After {{InlinerPass}}
; INL: IR Dump Before InlinerPass on (tester)
; INL: IR Dump After InlinerPass on (tester)

; INL-MOD: IR Dump Before InlinerPass on (tester, foo)
; INL-MOD: IR Dump After InlinerPass on (tester, foo) (invalidated)
; INL-MOD: IR Dump Before InlinerPass on (tester)
; INL-MOD: IR Dump After InlinerPass on (tester)


define void @tester() noinline {
  call void @foo()
  ret void
}

define internal void @foo() alwaysinline {
  call void @tester()
  ret void
}
