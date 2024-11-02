; RUN: not --crash llc < %s -mtriple=s390x-linux-gnu -argext-abi-check 2>&1 \
; RUN:   | FileCheck %s
; REQUIRES: asserts
;
; Test detection of missing extension involving an internal function which is
; passed as a function pointer to an external function.

define internal i32 @bar(i32 %Arg) {
  ret i32 %Arg
}

declare void @ExtFun(ptr %FunPtr);

define void @foo() {
  call void @ExtFun(ptr @bar)
  ret void
}

; CHECK: Narrow integer argument must have a valid extension type
