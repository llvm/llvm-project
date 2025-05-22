; Check that when removing arguments, existing callsite attributes are preserved

; RUN: llvm-reduce --abort-on-invalid-reduction --delta-passes=arguments --test FileCheck --test-arg --check-prefixes=INTERESTING --test-arg %s --test-arg --input-file %s -o %t
; RUN: FileCheck --check-prefixes=RESULT %s < %t

; INTERESTING-LABEL: define void @callee0(
define void @callee0(ptr %interesting0, ptr %interesting1, i32 %uninteresting2) {
  ret void
}

; INTERESTING-LABEL: define void @callee1(
define void @callee1(ptr byval(i64) %interesting0, ptr %interesting1, i32 %uninteresting2) {
  ret void
}

; INTERESTING-LABEL: define void @caller0(

; INTERESTING: byval
; INTERESTING-SAME: "some-attr"

; INTERESTING: byval
; INTERESTING-SAME: "more-attr"

; RESULT-LABEL: define void @caller0(ptr %val0) {
; RESULT: call void @callee0(ptr byval(i32) %val0, ptr "some-attr" %alloca0) #0
; RESULT: call void @callee1(ptr byval(i64) %alloca1, ptr "more-attr" %alloca1) #1
define void @caller0(ptr %val0, i32 %val1) {
  %alloca0 = alloca i32
  %alloca1 = alloca i64
  call void @callee0(ptr byval(i32) %val0, ptr "some-attr" %alloca0, i32 %val1) nounwind memory(none) "a-func-attr"
  call void @callee1(ptr byval(i64) %alloca1, ptr "more-attr" %alloca1, i32 9) "val-func-attr="="something"
  ret void
}

; RESULT-LABEL: define ptr @callee2() {
; RESULT-NEXT: ret ptr null
define ptr @callee2(ptr %val0, i32 %b) {
  store i32 %b, ptr %val0
  ret ptr %val0
}

; Make sure ret attributes are preserved
; INTERESTING: define ptr @caller1(
; INTERESTING: call

; RESULT-LABEL: define ptr @caller1() {
; RESULT: %ret = call align 4 "ret-attr" ptr @callee2()

define ptr @caller1(ptr %val0, i32 %val1) {
  %ret = call align 4 "ret-attr" ptr @callee2(ptr %val0, i32 %val1)
  ret ptr %ret
}

; RESULT: attributes #0 = { nounwind memory(none) "a-func-attr" }
; RESULT: attributes #1 = { "val-func-attr="="something" }
