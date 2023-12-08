; RUN: opt -passes=ipsccp < %s -S | FileCheck %s

define internal {i32, i32} @identity(i32 %patatino) {
  %foo = insertvalue {i32, i32} {i32 1, i32 undef}, i32 %patatino, 1
  ret {i32, i32} %foo
}

; Check that the return value is not transformed to undef
; CHECK: define internal { i32, i32 } @identity(i32 %patatino) {
; CHECK-NEXT:  %foo = insertvalue { i32, i32 } { i32 1, i32 undef }, i32 %patatino, 1
; CHECK-NEXT:  ret { i32, i32 } %foo
; CHECK-NEXT: }


define {i32, i32} @caller(i32 %pat) {
  %S1 = call {i32, i32} @identity(i32 %pat)
  ret {i32, i32} %S1
}

; Check that we don't invent values and propagate them.
; CHECK: define { i32, i32 } @caller(i32 %pat) {
; CHECK-NEXT:  %S1 = call { i32, i32 } @identity(i32 %pat)
; CHECK-NEXT:  ret { i32, i32 } %S1
; CHECK-NEXT: }
