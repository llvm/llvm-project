; RUN: opt < %s -basic-aa -aa-eval -print-all-alias-modref-info 2>&1 | FileCheck %s

; When merging MustAlias and PartialAlias, merge to PartialAlias
; instead of MayAlias.


target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"

; CHECK: PartialAlias:  i16* %bigbase0, i8* %phi
define i8 @test0(i8* %base, i1 %x) {
entry:
  %baseplusone = getelementptr i8, i8* %base, i64 1
  br i1 %x, label %red, label %green
red:
  br label %green
green:
  %phi = phi i8* [ %baseplusone, %red ], [ %base, %entry ]
  store i8 0, i8* %phi

  %bigbase0 = bitcast i8* %base to i16*
  store i16 -1, i16* %bigbase0

  %loaded = load i8, i8* %phi
  ret i8 %loaded
}

; CHECK: PartialAlias:  i16* %bigbase1, i8* %sel
define i8 @test1(i8* %base, i1 %x) {
entry:
  %baseplusone = getelementptr i8, i8* %base, i64 1
  %sel = select i1 %x, i8* %baseplusone, i8* %base
  store i8 0, i8* %sel

  %bigbase1 = bitcast i8* %base to i16*
  store i16 -1, i16* %bigbase1

  %loaded = load i8, i8* %sel
  ret i8 %loaded
}
