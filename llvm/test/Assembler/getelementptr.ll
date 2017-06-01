; RUN: llvm-as < %s | llvm-dis | llvm-as | llvm-dis | FileCheck %s
; RUN: verify-uselistorder %s

; Verify that over-indexed getelementptrs are folded.
@A = external global [2 x [3 x [5 x [7 x i32]]]]
@B = global i32* getelementptr ([2 x [3 x [5 x [7 x i32]]]], [2 x [3 x [5 x [7 x i32]]]]* @A, i64 0, i64 0, i64 2, i64 1, i64 7523)
; CHECK: @B = global i32* getelementptr ([2 x [3 x [5 x [7 x i32]]]], [2 x [3 x [5 x [7 x i32]]]]* @A, i64 36, i64 0, i64 1, i64 0, i64 5)
@C = global i32* getelementptr ([2 x [3 x [5 x [7 x i32]]]], [2 x [3 x [5 x [7 x i32]]]]* @A, i64 3, i64 2, i64 0, i64 0, i64 7523)
; CHECK: @C = global i32* getelementptr ([2 x [3 x [5 x [7 x i32]]]], [2 x [3 x [5 x [7 x i32]]]]* @A, i64 39, i64 1, i64 1, i64 4, i64 5)

; Verify that constant expression GEPs work with i84 indices.
@D = external global [1 x i32]

@E = global i32* getelementptr inbounds ([1 x i32], [1 x i32]* @D, i84 0, i64 1)
; CHECK: @E = global i32* getelementptr inbounds ([1 x i32], [1 x i32]* @D, i84 1, i64 0)

; Verify that i16 indices work.
@x = external global {i32, i32}
@y = global i32* getelementptr ({ i32, i32 }, { i32, i32 }* @x, i16 42, i32 0)
; CHECK: @y = global i32* getelementptr ({ i32, i32 }, { i32, i32 }* @x, i16 42, i32 0)

@PR23753_a = external global i8
@PR23753_b = global i8* getelementptr (i8, i8* @PR23753_a, i64 ptrtoint (i8* @PR23753_a to i64))
; CHECK: @PR23753_b = global i8* getelementptr (i8, i8* @PR23753_a, i64 ptrtoint (i8* @PR23753_a to i64))

; Verify that inrange on an index inhibits over-indexed getelementptr folding.

@nestedarray = global [2 x [4 x i8*]] zeroinitializer

; CHECK: @nestedarray.1 = alias i8*, getelementptr inbounds ([2 x [4 x i8*]], [2 x [4 x i8*]]* @nestedarray, inrange i32 0, i64 1, i32 0)
@nestedarray.1 = alias i8*, getelementptr inbounds ([2 x [4 x i8*]], [2 x [4 x i8*]]* @nestedarray, inrange i32 0, i32 0, i32 4)

; CHECK: @nestedarray.2 = alias i8*, getelementptr inbounds ([2 x [4 x i8*]], [2 x [4 x i8*]]* @nestedarray, i32 0, inrange i32 0, i32 4)
@nestedarray.2 = alias i8*, getelementptr inbounds ([2 x [4 x i8*]], [2 x [4 x i8*]]* @nestedarray, i32 0, inrange i32 0, i32 4)

; CHECK: @nestedarray.3 = alias i8*, getelementptr inbounds ([2 x [4 x i8*]], [2 x [4 x i8*]]* @nestedarray, i32 0, inrange i32 0, i32 0)
@nestedarray.3 = alias i8*, getelementptr inbounds ([4 x i8*], [4 x i8*]* getelementptr inbounds ([2 x [4 x i8*]], [2 x [4 x i8*]]* @nestedarray, i32 0, inrange i32 0), i32 0, i32 0)

; CHECK: @nestedarray.4 = alias i8*, getelementptr inbounds ([2 x [4 x i8*]], [2 x [4 x i8*]]* @nestedarray, i32 0, i32 1, i32 0)
@nestedarray.4 = alias i8*, getelementptr inbounds ([4 x i8*], [4 x i8*]* getelementptr inbounds ([2 x [4 x i8*]], [2 x [4 x i8*]]* @nestedarray, i32 0, inrange i32 0), i32 1, i32 0)

; CHECK: @nestedarray.5 = alias i8*, getelementptr inbounds ([2 x [4 x i8*]], [2 x [4 x i8*]]* @nestedarray, inrange i32 0, i32 1, i32 0)
@nestedarray.5 = alias i8*, getelementptr inbounds ([4 x i8*], [4 x i8*]* getelementptr inbounds ([2 x [4 x i8*]], [2 x [4 x i8*]]* @nestedarray, inrange i32 0, i32 0), i32 1, i32 0)

; See if i92 indices work too.
define i32 *@test({i32, i32}* %t, i92 %n) {
; CHECK: @test
; CHECK: %B = getelementptr { i32, i32 }, { i32, i32 }* %t, i92 %n, i32 0
  %B = getelementptr {i32, i32}, {i32, i32}* %t, i92 %n, i32 0
  ret i32* %B
}

; Verify that constant expression vector GEPs work.

@z = global <2 x i32*> getelementptr ([3 x {i32, i32}], <2 x [3 x {i32, i32}]*> zeroinitializer, <2 x i32> <i32 1, i32 2>, <2 x i32> <i32 2, i32 3>, <2 x i32> <i32 1, i32 1>)

; Verify that struct GEP works with a vector of pointers.
define <2 x i32*> @test7(<2 x {i32, i32}*> %a) {
  %w = getelementptr {i32, i32}, <2 x {i32, i32}*> %a, <2 x i32> <i32 5, i32 9>, <2 x i32> zeroinitializer
  ret <2 x i32*> %w
}

; Verify that array GEP works with a vector of pointers.
define <2 x i8*> @test8(<2 x [2 x i8]*> %a) {
  %w = getelementptr  [2 x i8], <2 x  [2 x i8]*> %a, <2 x i32> <i32 0, i32 0>, <2 x i8> <i8 0, i8 1>
  ret <2 x i8*> %w
}

@array = internal global [16 x i32] [i32 -200, i32 -199, i32 -198, i32 -197, i32 -196, i32 -195, i32 -194, i32 -193, i32 -192, i32 -191, i32 -190, i32 -189, i32 -188, i32 -187, i32 -186, i32 -185], align 16

; Verify that array GEP doesn't incorrectly infer inbounds.
define i32* @test9() {
entry:
  ret i32* getelementptr ([16 x i32], [16 x i32]* @array, i64 0, i64 -13)
; CHECK-LABEL: define i32* @test9(
; CHECK: ret i32* getelementptr ([16 x i32], [16 x i32]* @array, i64 0, i64 -13)
}
