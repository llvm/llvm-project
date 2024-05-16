; RUN: opt < %s -passes=debugify,instcombine -S | FileCheck %s
; RUN: opt < %s -passes=debugify,instcombine -S --try-experimental-debuginfo-iterators | FileCheck %s

declare i32 @escape(i32)

; CHECK-LABEL: define {{.*}}@foo(
define i32 @foo(i1 %c1) {
entry:
  %baz = alloca i32
  br i1 %c1, label %lhs, label %rhs

lhs:
  store i32 1, ptr %baz
  br label %cleanup

rhs:
  store i32 2, ptr %baz
  br label %cleanup

cleanup:
  ; CHECK: %storemerge = phi i32 [ 2, %rhs ], [ 1, %lhs ], !dbg [[merge_loc:![0-9]+]]
  %baz.val = load i32, ptr %baz
  %ret.val = call i32 @escape(i32 %baz.val)
  ret i32 %ret.val
}

; CHECK: [[merge_loc]] = !DILocation(line: 0
