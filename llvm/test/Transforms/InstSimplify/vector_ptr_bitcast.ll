; RUN: opt -S -instsimplify < %s | FileCheck %s
target datalayout = "e-i64:64-f80:128-n8:16:32:64-S128"

%mst = type { i8*, i8* }
%mst2 = type { i32*, i32*, i32*, i32* }

@a = private unnamed_addr constant %mst { i8* inttoptr (i64 -1 to i8*),
                                          i8* inttoptr (i64 -1 to i8*)},
                                          align 8
@b = private unnamed_addr constant %mst2 { i32* inttoptr (i64 42 to i32*),
                                           i32* inttoptr (i64 67 to i32*),
                                           i32* inttoptr (i64 33 to i32*),
                                           i32* inttoptr (i64 58 to i32*)},
                                          align 8

define i64 @fn() {
  %x = load <2 x i8*>, <2 x i8*>* bitcast (%mst* @a to <2 x i8*>*), align 8
  %b = extractelement <2 x i8*> %x, i32 0
  %c = ptrtoint i8* %b to i64
  ; CHECK-LABEL: @fn
  ; CHECK-NEXT: ret i64 -1
  ret i64 %c
}

define i64 @fn2() {
  %x = load <4 x i32*>, <4 x i32*>* bitcast (%mst2* @b to <4 x i32*>*), align 8
  %b = extractelement <4 x i32*> %x, i32 0
  %c = extractelement <4 x i32*> %x, i32 3
  %d = ptrtoint i32* %b to i64
  %e = ptrtoint i32* %c to i64
  %r = add i64 %d, %e
  ; CHECK-LABEL: @fn2
  ; CHECK-NEXT: ret i64 100
  ret i64 %r
}
