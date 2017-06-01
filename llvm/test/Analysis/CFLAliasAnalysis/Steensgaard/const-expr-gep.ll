; This testcase consists of alias relations which should be completely
; resolvable by cfl-steens-aa, but require analysis of getelementptr constant exprs.
; Derived from BasicAA/2003-12-11-ConstExprGEP.ll

; RUN: opt < %s -disable-basicaa -cfl-steens-aa -aa-eval -print-may-aliases -disable-output 2>&1 | FileCheck %s

%T = type { i32, [10 x i8] }

@G = external global %T
@G2 = external global %T

; TODO: Quite a few of these are MayAlias because we don't yet consider
; constant offsets in CFLSteensAA. If we start doing so, then we'll need to
; change these test cases

; CHECK:     Function: test
; CHECK:     MayAlias: i32* %D, i32* %F
; CHECK:     MayAlias: i32* %D, i8* %X
; CHECK:     MayAlias: i32* %F, i8* %X
define void @test() {
  %D = getelementptr %T, %T* @G, i64 0, i32 0
  %F = getelementptr i32, i32* getelementptr (%T, %T* @G, i64 0, i32 0), i64 0
  %X = getelementptr [10 x i8], [10 x i8]* getelementptr (%T, %T* @G, i64 0, i32 1), i64 0, i64 5

  ret void
}

; CHECK:     Function: simplecheck
; CHECK:     MayAlias: i32* %F, i32* %arg0
; CHECK:     MayAlias: i32* %H, i32* %arg0
; CHECK:     MayAlias: i32* %F, i32* %H
define void @simplecheck(i32* %arg0) {
  %F = getelementptr i32, i32* getelementptr (%T, %T* @G, i64 0, i32 0), i64 0
  %H = getelementptr %T, %T* @G2, i64 0, i32 0

  ret void
}

; Ensure that CFLSteensAA properly identifies and handles escaping variables (i.e.
; globals) in nested ConstantExprs

; CHECK:      Function: checkNesting
; CHECK:      MayAlias: i32* %A, i32* %arg0

%NestedT = type { [1 x [1 x i32]] }
@NT = external global %NestedT

define void @checkNesting(i32* %arg0) {
  %A = getelementptr [1 x i32],
         [1 x i32]* getelementptr
           ([1 x [1 x i32]], [1 x [1 x i32]]* getelementptr (%NestedT, %NestedT* @NT, i64 0, i32 0),
           i64 0,
           i32 0),
         i64 0,
         i32 0
  ret void
}
