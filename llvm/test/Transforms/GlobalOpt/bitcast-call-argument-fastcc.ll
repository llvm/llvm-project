; RUN:  opt < %s -globalopt -S | FileCheck %s

; Check that globalopt doesn't set fastcc on
; functions and their call sites if its hasAddressTaken()
; is true. "foo" function's hasAddressTaken() is
; true so globalopt should not set fastcc on it
; and its call sites. However, for "bar" fastcc must be set.

; CHECK-NOT: define internal fastcc i32 @foo() {
; CHECK:     define internal i32 @foo() {
define internal i32 @foo() {
entry:
    ret i32 5
}

; CHECK: define internal fastcc i32 @bar(float ()* %arg) unnamed_addr {
define internal i32 @bar(float()* %arg) {
    ret i32 5
}

; CHECK-NOT: define fastcc i32 @test()
; CHECK:     define i32 @test()
define i32 @test() {
  ; CHECK: call fastcc i32 @bar(float ()* bitcast (i32 ()* @foo to float ()*))
  %v1 = call i32 @bar(float ()* bitcast (i32 ()* @foo to float ()*))
  %v2 = add i32 %v1, 6
  ret i32 %v2
}

