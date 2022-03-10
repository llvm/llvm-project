; This testcase ensures that CFL AA responds conservatively when we union 
; groups of pointers together through ternary/conditional operations
; Derived from:
; void foo(bool c) {
;   char a, b;
;   char *m = c ? &a : &b;
;   *m;
; }
;

; RUN: opt < %s -aa-pipeline=cfl-steens-aa -passes=aa-eval -print-may-aliases -disable-output 2>&1 | FileCheck %s

%T = type { i32, [10 x i8] }

; CHECK:     Function: test

define void @test(i1 %C) {
; CHECK: 10 Total Alias Queries Performed
; CHECK: 4 no alias responses
  %M = alloca %T*, align 8 ; NoAlias with %A, %B, %MS, %AP
  %A = alloca %T, align 8
  %B = alloca %T, align 8

  %MS = select i1 %C, %T* %B, %T* %A

  store %T* %MS, %T** %M

  %AP = load %T*, %T** %M ; PartialAlias with %A, %B

  ret void
}
