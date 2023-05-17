; RUN: opt -mtriple=arm64-darwin-unknown -S -passes=consthoist < %s | FileCheck %s

%T = type { i32, i32, i32, i32 }

define i32 @test1() nounwind {
; CHECK-LABEL: test1
; CHECK: %const = bitcast i64 68141056 to i64
; CHECK: %1 = inttoptr i64 %const to ptr
; CHECK: %o1 = getelementptr %T, ptr %1, i32 0, i32 1
; CHECK: %o2 = getelementptr %T, ptr %1, i32 0, i32 2
; CHECK: %o3 = getelementptr %T, ptr %1, i32 0, i32 3
  %at = inttoptr i64 68141056 to ptr
  %o1 = getelementptr %T, ptr %at, i32 0, i32 1
  %t1 = load i32, ptr %o1
  %o2 = getelementptr %T, ptr %at, i32 0, i32 2
  %t2 = load i32, ptr %o2
  %a1 = add i32 %t1, %t2
  %o3 = getelementptr %T, ptr %at, i32 0, i32 3
  %t3 = load i32, ptr %o3
  %a2 = add i32 %a1, %t3
  ret i32 %a2
}

