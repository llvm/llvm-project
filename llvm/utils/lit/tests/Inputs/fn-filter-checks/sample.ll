; REQUIRES: opt
;
; RUN: opt -S < %s -passes=instcombine | FileCheck %s

define i32 @foo(i32 %x) {
  ret i32 %x
}

define i32 @bar(i32 %x) {
  ret i32 %x
}

; CHECK-LABEL: @foo
; CHECK: ret i32 %x

; CHECK-LABEL: @bar
; CHECK: ret i32 FILTER_AWAY
