; RUN: llvm-reduce %s -o %t --abort-on-invalid-reduction --delta-passes=operands-to-args --test FileCheck --test-arg %s --test-arg --check-prefix=INTERESTING --test-arg --input-file
; RUN: FileCheck %s --input-file %t --check-prefix=REDUCED

@a = dso_local global i8 0, align 1
@b = dso_local global i16 0, align 2


declare ptr @callee(ptr %a, i16)

; INTERESTING-LABEL: define ptr @caller(
; INTERESTING: sext
; INTERESTING: icmp

; REDUCED-LABEL: define ptr @caller(ptr %some.ptr, ptr %a, i8 %ld0, ptr %b, i16 %ld1, i32 %conv, i32 %conv1, i1 %cmp, i16 %conv2, ptr %callee.ret) #0 {

; REDUCED: %callee.ret8 = call align 8 ptr @callee(ptr align 8 "some-attr" %some.ptr, i16 signext %conv2) #1

define ptr @caller(ptr %some.ptr) nounwind {
entry:
  %ld0 = load i8, ptr @a, align 1
  %conv = zext i8 %ld0 to i32
  %ld1 = load i16, ptr @b, align 2
  %conv1 = sext i16 %ld1 to i32
  %cmp = icmp sge i32 %conv, %conv1
  %conv2 = sext i1 %cmp to i16
  %callee.ret = call align 8 ptr @callee(ptr align 8 "some-attr" %some.ptr, i16 signext %conv2) nocallback
  ret ptr %callee.ret
}

; REDUCED: attributes #0 = { nounwind }
; REDUCED: attributes #1 = { nocallback }
