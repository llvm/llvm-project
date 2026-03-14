; RUN: opt -aa-pipeline=basic-aa -passes=aa-eval -print-all-alias-modref-info -disable-output < %s 2>&1 | FileCheck %s

; A call return value is not always an escape source, because
; CaptureTracking can look through some calls. The test is constructed to
; hit the getUnderlyingObject() recursion limit.
define i32 @test() {
; CHECK-LABEL: Function: test
; CHECK-NEXT: MustAlias: i32* %a, i32* %p7
  %a = alloca i32
  %p1 = call ptr @llvm.strip.invariant.group.p0(ptr %a)
  %p2 = getelementptr i8, ptr %p1, i64 1
  %p3 = getelementptr i8, ptr %p2, i64 -1
  %p4 = getelementptr i8, ptr %p3, i64 1
  %p5 = getelementptr i8, ptr %p4, i64 -1
  %p6 = getelementptr i8, ptr %p5, i64 1
  %p7 = getelementptr i8, ptr %p6, i64 -1
  %v = load i32, ptr %a
  store i32 -1, ptr %p7
  ret i32 %v
}

declare ptr @llvm.strip.invariant.group.p0(ptr)
