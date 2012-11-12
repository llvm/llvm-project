; RUN: llvm-as < %s | llvm-dis | llvm-as | llvm-dis | FileCheck %s

; Verify that over-indexed getelementptrs are folded.
@A = external global [2 x [3 x [5 x [7 x i32]]]]
@B = global i32* getelementptr ([2 x [3 x [5 x [7 x i32]]]]* @A, i64 0, i64 0, i64 2, i64 1, i64 7523)
; CHECK: @B = global i32* getelementptr ([2 x [3 x [5 x [7 x i32]]]]* @A, i64 36, i64 0, i64 1, i64 0, i64 5)
@C = global i32* getelementptr ([2 x [3 x [5 x [7 x i32]]]]* @A, i64 3, i64 2, i64 0, i64 0, i64 7523)
; CHECK: @C = global i32* getelementptr ([2 x [3 x [5 x [7 x i32]]]]* @A, i64 39, i64 1, i64 1, i64 4, i64 5)

;; Verify that i16 indices work.
@x = external global {i32, i32}
@y = global i32* getelementptr ({ i32, i32 }* @x, i16 42, i32 0)
; CHECK: @y = global i32* getelementptr ({ i32, i32 }* @x, i16 42, i32 0)

; see if i92 indices work too.
define i32 *@test({i32, i32}* %t, i92 %n) {
; CHECK: @test
; CHECK: %B = getelementptr { i32, i32 }* %t, i92 %n, i32 0
  %B = getelementptr {i32, i32}* %t, i92 %n, i32 0
  ret i32* %B
}

