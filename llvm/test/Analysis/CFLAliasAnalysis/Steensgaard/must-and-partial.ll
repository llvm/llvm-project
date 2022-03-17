; RUN: opt < %s -aa-pipeline=cfl-steens-aa -passes=aa-eval -print-all-alias-modref-info 2>&1 | FileCheck %s
; When merging MustAlias and PartialAlias, merge to PartialAlias
; instead of MayAlias.


target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"

; FIXME: This could be PartialAlias but CFLSteensAA can't currently prove it
; CHECK: MayAlias:  i16* %bigbase0, i8* %phi
define i8 @test0(i1 %x) {
entry:
  %base = alloca i8, align 4
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

; FIXME: This could be PartialAlias but CFLSteensAA can't currently prove it
; CHECK: MayAlias:  i16* %bigbase1, i8* %sel
define i8 @test1(i1 %x) {
entry:
  %base = alloca i8, align 4
  %baseplusone = getelementptr i8, i8* %base, i64 1
  %sel = select i1 %x, i8* %baseplusone, i8* %base
  store i8 0, i8* %sel

  %bigbase1 = bitcast i8* %base to i16*
  store i16 -1, i16* %bigbase1

  %loaded = load i8, i8* %sel
  ret i8 %loaded
}

; Incoming pointer arguments should not be MayAlias because we do not know their initial state
; even if they are nocapture
; CHECK: MayAlias:  double* %A, double* %Index
define void @testr2(double* nocapture readonly %A, double* nocapture readonly %Index) {
  %arrayidx22 = getelementptr inbounds double, double* %Index, i64 2
  %1 = load double, double* %arrayidx22
  %arrayidx25 = getelementptr inbounds double, double* %A, i64 2
  %2 = load double, double* %arrayidx25
  %3 = fneg double %1
  %mul26 = fmul double %3, %2
  load double, double* %A
  load double, double* %Index
  ret void
}
