; RUN: opt < %s -aa-pipeline=basic-aa -passes=aa-eval -print-all-alias-modref-info 2>&1 | FileCheck %s

; When merging MustAlias and PartialAlias, merge to PartialAlias
; instead of MayAlias.


target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"

; CHECK: PartialAlias:  i16* %base, i8* %phi
define i8 @test0(ptr %base, i1 %x) {
entry:
  %baseplusone = getelementptr i8, ptr %base, i64 1
  br i1 %x, label %red, label %green
red:
  br label %green
green:
  %phi = phi ptr [ %baseplusone, %red ], [ %base, %entry ]
  store i8 0, ptr %phi

  store i16 -1, ptr %base

  %loaded = load i8, ptr %phi
  ret i8 %loaded
}

; CHECK: PartialAlias:  i16* %base, i8* %sel
define i8 @test1(ptr %base, i1 %x) {
entry:
  %baseplusone = getelementptr i8, ptr %base, i64 1
  %sel = select i1 %x, ptr %baseplusone, ptr %base
  store i8 0, ptr %sel

  store i16 -1, ptr %base

  %loaded = load i8, ptr %sel
  ret i8 %loaded
}
