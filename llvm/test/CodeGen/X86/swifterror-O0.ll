; RUN: llc -mtriple=x86_64-apple-darwin -O0 %s -o - | FileCheck %s

define i32 @foo() {
; CHECK-LABEL: foo:
; CHECK: cmpq $0, %r12

  %swifterror = alloca swifterror i8*
  call void @callee(i8** swifterror %swifterror)
  %err = load i8*, i8** %swifterror
  %tst = icmp ne i8* %err, null
  br i1 %tst, label %true, label %false

true:
  ret i32 0

false:
  ret i32 1
}

declare void @callee(i8** swifterror)
