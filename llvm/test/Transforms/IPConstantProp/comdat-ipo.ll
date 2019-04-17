; RUN: opt < %s -ipconstprop -S | FileCheck %s

; See PR26774

define i32 @baz() {
  ret i32 10
}

; We can const-prop @baz's return value *into* @foo, but cannot
; constprop @foo's return value into bar.

define linkonce_odr i32 @foo() {
; CHECK-LABEL: @foo(
; CHECK-NEXT:  %val = call i32 @baz()
; CHECK-NEXT:  ret i32 10

  %val = call i32 @baz()
  ret i32 %val
}

define i32 @bar() {
; CHECK-LABEL: @bar(
; CHECK-NEXT:  %val = call i32 @foo()
; CHECK-NEXT:  ret i32 %val

  %val = call i32 @foo()
  ret i32 %val
}
