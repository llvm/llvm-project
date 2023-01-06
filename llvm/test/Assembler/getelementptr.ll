; RUN: llvm-as  < %s | llvm-dis  | llvm-as  | llvm-dis  | FileCheck %s
; RUN: verify-uselistorder  %s

; Verify that over-indexed getelementptrs are folded.
@A = external global [2 x [3 x [5 x [7 x i32]]]]
@B = global ptr getelementptr ([2 x [3 x [5 x [7 x i32]]]], ptr @A, i64 0, i64 0, i64 2, i64 1, i64 7523)
; CHECK: @B = global ptr getelementptr ([2 x [3 x [5 x [7 x i32]]]], ptr @A, i64 36, i64 0, i64 1, i64 0, i64 5)
@C = global ptr getelementptr ([2 x [3 x [5 x [7 x i32]]]], ptr @A, i64 3, i64 2, i64 0, i64 0, i64 7523)
; CHECK: @C = global ptr getelementptr ([2 x [3 x [5 x [7 x i32]]]], ptr @A, i64 39, i64 1, i64 1, i64 4, i64 5)

; Verify that constant expression GEPs work with i84 indices.
@D = external global [1 x i32]

@E = global ptr getelementptr inbounds ([1 x i32], ptr @D, i84 0, i64 1)
; CHECK: @E = global ptr getelementptr inbounds ([1 x i32], ptr @D, i84 1, i64 0)

; Verify that i16 indices work.
@x = external global {i32, i32}
@y = global ptr getelementptr ({ i32, i32 }, ptr @x, i16 42, i32 0)
; CHECK: @y = global ptr getelementptr ({ i32, i32 }, ptr @x, i16 42, i32 0)

@PR23753_a = external global i8
@PR23753_b = global ptr getelementptr (i8, ptr @PR23753_a, i64 ptrtoint (ptr @PR23753_a to i64))
; CHECK: @PR23753_b = global ptr getelementptr (i8, ptr @PR23753_a, i64 ptrtoint (ptr @PR23753_a to i64))

; Verify that inrange on an index inhibits over-indexed getelementptr folding.

@nestedarray = global [2 x [4 x ptr]] zeroinitializer

; CHECK: @nestedarray.1 = alias ptr, getelementptr inbounds ([2 x [4 x ptr]], ptr @nestedarray, inrange i32 0, i64 1, i32 0)
@nestedarray.1 = alias ptr, getelementptr inbounds ([2 x [4 x ptr]], ptr @nestedarray, inrange i32 0, i32 0, i32 4)

; CHECK: @nestedarray.2 = alias ptr, getelementptr inbounds ([2 x [4 x ptr]], ptr @nestedarray, i32 0, inrange i32 0, i32 4)
@nestedarray.2 = alias ptr, getelementptr inbounds ([2 x [4 x ptr]], ptr @nestedarray, i32 0, inrange i32 0, i32 4)

; CHECK: @nestedarray.3 = alias ptr, getelementptr inbounds ([2 x [4 x ptr]], ptr @nestedarray, i32 0, inrange i32 0)
@nestedarray.3 = alias ptr, getelementptr inbounds ([4 x ptr], ptr getelementptr inbounds ([2 x [4 x ptr]], ptr @nestedarray, i32 0, inrange i32 0), i32 0, i32 0)

; CHECK: @nestedarray.4 = alias ptr, getelementptr inbounds ([2 x [4 x ptr]], ptr @nestedarray, i32 0, i32 1, i32 0)
@nestedarray.4 = alias ptr, getelementptr inbounds ([4 x ptr], ptr getelementptr inbounds ([2 x [4 x ptr]], ptr @nestedarray, i32 0, inrange i32 0), i32 1, i32 0)

; CHECK: @nestedarray.5 = alias ptr, getelementptr inbounds ([2 x [4 x ptr]], ptr @nestedarray, inrange i32 0, i32 1, i32 0)
@nestedarray.5 = alias ptr, getelementptr inbounds ([4 x ptr], ptr getelementptr inbounds ([2 x [4 x ptr]], ptr @nestedarray, inrange i32 0, i32 0), i32 1, i32 0)

; See if i92 indices work too.
define ptr @test(ptr %t, i92 %n) {
; CHECK: @test
; CHECK: %B = getelementptr { i32, i32 }, ptr %t, i92 %n, i32 0
  %B = getelementptr {i32, i32}, ptr %t, i92 %n, i32 0
  ret ptr %B
}

; Verify that constant expression vector GEPs work.

@z = global <2 x ptr> getelementptr ([3 x {i32, i32}], <2 x ptr> zeroinitializer, <2 x i32> <i32 1, i32 2>, <2 x i32> <i32 2, i32 3>, <2 x i32> <i32 1, i32 1>)

; Verify that struct GEP works with a vector of pointers.
define <2 x ptr> @test7(<2 x ptr> %a) {
  %w = getelementptr {i32, i32}, <2 x ptr> %a, <2 x i32> <i32 5, i32 9>, <2 x i32> zeroinitializer
  ret <2 x ptr> %w
}

; Verify that array GEP works with a vector of pointers.
define <2 x ptr> @test8(<2 x ptr> %a) {
  %w = getelementptr  [2 x i8], <2 x  ptr> %a, <2 x i32> <i32 0, i32 0>, <2 x i8> <i8 0, i8 1>
  ret <2 x ptr> %w
}

@array = internal global [16 x i32] [i32 -200, i32 -199, i32 -198, i32 -197, i32 -196, i32 -195, i32 -194, i32 -193, i32 -192, i32 -191, i32 -190, i32 -189, i32 -188, i32 -187, i32 -186, i32 -185], align 16

; Verify that array GEP doesn't incorrectly infer inbounds.
define ptr @test9() {
entry:
  ret ptr getelementptr ([16 x i32], ptr @array, i64 0, i64 -13)
; CHECK-LABEL: define ptr @test9(
; CHECK: ret ptr getelementptr ([16 x i32], ptr @array, i64 0, i64 -13)
}
