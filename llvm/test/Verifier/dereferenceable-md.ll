; RUN: not llvm-as < %s -o /dev/null 2>&1 | FileCheck %s

declare ptr @foo()

define void @f1() {
entry:
  call ptr @foo(), !dereferenceable !{i64 2}
  ret void
}
; CHECK: dereferenceable, dereferenceable_or_null apply only to load and inttoptr instructions, use attributes for calls or invokes
; CHECK-NEXT: call ptr @foo()

define void @f2() {
entry:
  call ptr @foo(), !dereferenceable_or_null !{i64 2}
  ret void
}
; CHECK: dereferenceable, dereferenceable_or_null apply only to load and inttoptr instructions, use attributes for calls or invokes
; CHECK-NEXT: call ptr @foo()

define i8 @f3(ptr %x) {
entry:
  %y = load i8, ptr %x, !dereferenceable !{i64 2}
  ret i8 %y
}
; CHECK: dereferenceable, dereferenceable_or_null apply only to pointer types
; CHECK-NEXT: load i8, ptr %x

define i8 @f4(ptr %x) {
entry:
  %y = load i8, ptr %x, !dereferenceable_or_null !{i64 2}
  ret i8 %y
}
; CHECK: dereferenceable, dereferenceable_or_null apply only to pointer types
; CHECK-NEXT: load i8, ptr %x

define ptr @f5(ptr %x) {
entry:
  %y = load ptr, ptr %x, !dereferenceable !{}
  ret ptr %y
}
; CHECK: dereferenceable, dereferenceable_or_null take one operand
; CHECK-NEXT: load ptr, ptr %x


define ptr @f6(ptr %x) {
entry:
  %y = load ptr, ptr %x, !dereferenceable_or_null !{}
  ret ptr %y
}
; CHECK: dereferenceable, dereferenceable_or_null take one operand
; CHECK-NEXT: load ptr, ptr %x

define ptr @f7(ptr %x) {
entry:
  %y = load ptr, ptr %x, !dereferenceable !{!"str"}
  ret ptr %y
}
; CHECK: dereferenceable, dereferenceable_or_null metadata value must be an i64!
; CHECK-NEXT: load ptr, ptr %x


define ptr @f8(ptr %x) {
entry:
  %y = load ptr, ptr %x, !dereferenceable_or_null !{!"str"}
  ret ptr %y
}
; CHECK: dereferenceable, dereferenceable_or_null metadata value must be an i64!
; CHECK-NEXT: load ptr, ptr %x

define ptr @f9(ptr %x) {
entry:
  %y = load ptr, ptr %x, !dereferenceable !{i32 2}
  ret ptr %y
}
; CHECK: dereferenceable, dereferenceable_or_null metadata value must be an i64!
; CHECK-NEXT: load ptr, ptr %x


define ptr @f10(ptr %x) {
entry:
  %y = load ptr, ptr %x, !dereferenceable_or_null !{i32 2}
  ret ptr %y
}
; CHECK: dereferenceable, dereferenceable_or_null metadata value must be an i64!
; CHECK-NEXT: load ptr, ptr %x

define ptr @f_11(i8 %val) {
  %ptr = inttoptr i8 %val to ptr, !dereferenceable !{i32 2}
  ret ptr %ptr
}
; CHECK: dereferenceable, dereferenceable_or_null metadata value must be an i64!
; CHECK-NEXT: %ptr = inttoptr i8 %val to ptr, !dereferenceable !3

define ptr @f_12(i8 %val) {
  %ptr = inttoptr i8 %val to ptr, !dereferenceable_or_null !{i32 2}
  ret ptr %ptr
}
; CHECK: dereferenceable, dereferenceable_or_null metadata value must be an i64!
; CHECK-NEXT: %ptr = inttoptr i8 %val to ptr, !dereferenceable_or_null !3

define ptr @f_13(i8 %val) {
  %ptr = inttoptr i8 %val to ptr, !dereferenceable !{}
  ret ptr %ptr
}
; CHECK: dereferenceable, dereferenceable_or_null take one operand
; CHECK-NEXT: %ptr = inttoptr i8 %val to ptr, !dereferenceable !1

define ptr @f_14(i8 %val) {
  %ptr = inttoptr i8 %val to ptr, !dereferenceable_or_null !{}
  ret ptr %ptr
}
; CHECK: dereferenceable, dereferenceable_or_null take one operand
; CHECK-NEXT: %ptr = inttoptr i8 %val to ptr, !dereferenceable_or_null !1

define ptr @f_15(i8 %val) {
  %ptr = inttoptr i8 %val to ptr, !dereferenceable !{!"str"}
  ret ptr %ptr
}
; CHECK: dereferenceable, dereferenceable_or_null metadata value must be an i64!
; CHECK-NEXT: %ptr = inttoptr i8 %val to ptr, !dereferenceable !2

define ptr @f_16(i8 %val) {
  %ptr = inttoptr i8 %val to ptr, !dereferenceable_or_null !{!"str"}
  ret ptr %ptr
}
; CHECK: dereferenceable, dereferenceable_or_null metadata value must be an i64!
; CHECK-NEXT: %ptr = inttoptr i8 %val to ptr, !dereferenceable_or_null !2
