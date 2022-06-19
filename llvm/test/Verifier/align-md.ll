; RUN: not llvm-as < %s -o /dev/null 2>&1 | FileCheck %s

declare i8* @foo()

define void @f1() {
entry:
  call i8* @foo(), !align !{i64 2}
  ret void
}
; CHECK: align applies only to load instructions
; CHECK-NEXT: call i8* @foo()

define i8 @f2(i8* %x) {
entry:
  %y = load i8, i8* %x, !align !{i64 2}
  ret i8 %y
}
; CHECK: align applies only to pointer types
; CHECK-NEXT: load i8, i8* %x

define i8* @f3(i8** %x) {
entry:
  %y = load i8*, i8** %x, !align !{}
  ret i8* %y
}
; CHECK: align takes one operand
; CHECK-NEXT: load i8*, i8** %x

define i8* @f4(i8** %x) {
entry:
  %y = load i8*, i8** %x, !align !{!"str"}
  ret i8* %y
}
; CHECK: align metadata value must be an i64!
; CHECK-NEXT: load i8*, i8** %x

define i8* @f5(i8** %x) {
entry:
  %y = load i8*, i8** %x, !align !{i32 2}
  ret i8* %y
}
; CHECK: align metadata value must be an i64!
; CHECK-NEXT: load i8*, i8** %x

define i8* @f6(i8** %x) {
entry:
  %y = load i8*, i8** %x, !align !{i64 3}
  ret i8* %y
}
; CHECK: align metadata value must be a power of 2!
; CHECK-NEXT: load i8*, i8** %x

define i8* @f7(i8** %x) {
entry:
  %y = load i8*, i8** %x, !align !{i64 8589934592}
  ret i8* %y
}
; CHECK: alignment is larger that implementation defined limit
; CHECK-NEXT: load i8*, i8** %x
