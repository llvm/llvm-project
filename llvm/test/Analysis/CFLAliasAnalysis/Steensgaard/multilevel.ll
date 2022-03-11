; This testcase ensures that CFL AA handles trivial cases with storing 
; pointers in pointers appropriately.
; Derived from:
; char a, b;
; char *m = &a, *n = &b;
; *m;
; *n;

; RUN: opt < %s -aa-pipeline=cfl-steens-aa -passes=aa-eval -print-may-aliases -disable-output 2>&1 | FileCheck %s

%T = type { i32, [10 x i8] }

; CHECK:     Function: test

define void @test() {
; CHECK: 15 Total Alias Queries Performed
; CHECK: 13 no alias responses
  %M = alloca %T*, align 8
  %N = alloca %T*, align 8
  %A = alloca %T, align 8
  %B = alloca %T, align 8

  store %T* %A, %T** %M
  store %T* %B, %T** %N

  %AP = load %T*, %T** %M ; PartialAlias with %A
  %BP = load %T*, %T** %N ; PartialAlias with %B

  ret void
}
