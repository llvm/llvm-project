; RUN: not llvm-as < %s -o /dev/null 2>&1 | FileCheck %s

declare ptr @foo()

define void @f1() {
entry:
  call ptr @foo(), !align !{i64 2}
  ret void
}
; CHECK: align applies only to load instructions
; CHECK-NEXT: call ptr @foo()

define i8 @f2(ptr %x) {
entry:
  %y = load i8, ptr %x, !align !{i64 2}
  ret i8 %y
}
; CHECK: align applies only to pointer types
; CHECK-NEXT: load i8, ptr %x

define ptr @f3(ptr %x) {
entry:
  %y = load ptr, ptr %x, !align !{}
  ret ptr %y
}
; CHECK: align takes one operand
; CHECK-NEXT: load ptr, ptr %x

define ptr @f4(ptr %x) {
entry:
  %y = load ptr, ptr %x, !align !{!"str"}
  ret ptr %y
}
; CHECK: align metadata value must be an i64!
; CHECK-NEXT: load ptr, ptr %x

define ptr @f5(ptr %x) {
entry:
  %y = load ptr, ptr %x, !align !{i32 2}
  ret ptr %y
}
; CHECK: align metadata value must be an i64!
; CHECK-NEXT: load ptr, ptr %x

define ptr @f6(ptr %x) {
entry:
  %y = load ptr, ptr %x, !align !{i64 3}
  ret ptr %y
}
; CHECK: align metadata value must be a power of 2!
; CHECK-NEXT: load ptr, ptr %x

define ptr @f7(ptr %x) {
entry:
  %y = load ptr, ptr %x, !align !{i64 8589934592}
  ret ptr %y
}
; CHECK: alignment is larger that implementation defined limit
; CHECK-NEXT: load ptr, ptr %x
