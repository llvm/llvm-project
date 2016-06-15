; RUN: opt < %s -basicaa -slp-vectorizer -S |FileCheck %s
; RUN: opt < %s -aa-pipeline=basic-aa -passes=slp-vectorizer -S |FileCheck %s
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-unknown"

; Test if SLP can handle GEP expressions.
; The test perform the following action:
;   x->first  = y->first  + 16
;   x->second = y->second + 16

; CHECK-LABEL: foo1
; CHECK: <2 x i32*>
define void @foo1 ({ i32*, i32* }* noalias %x, { i32*, i32* }* noalias %y) {
  %1 = getelementptr inbounds { i32*, i32* }, { i32*, i32* }* %y, i64 0, i32 0
  %2 = load i32*, i32** %1, align 8
  %3 = getelementptr inbounds i32, i32* %2, i64 16
  %4 = getelementptr inbounds { i32*, i32* }, { i32*, i32* }* %x, i64 0, i32 0
  store i32* %3, i32** %4, align 8
  %5 = getelementptr inbounds { i32*, i32* }, { i32*, i32* }* %y, i64 0, i32 1
  %6 = load i32*, i32** %5, align 8
  %7 = getelementptr inbounds i32, i32* %6, i64 16
  %8 = getelementptr inbounds { i32*, i32* }, { i32*, i32* }* %x, i64 0, i32 1
  store i32* %7, i32** %8, align 8
  ret void
}

; Test that we don't vectorize GEP expressions if indexes are not constants.
; We can't produce an efficient code in that case.
; CHECK-LABEL: foo2
; CHECK-NOT: <2 x i32*>
define void @foo2 ({ i32*, i32* }* noalias %x, { i32*, i32* }* noalias %y, i32 %i) {
  %1 = getelementptr inbounds { i32*, i32* }, { i32*, i32* }* %y, i64 0, i32 0
  %2 = load i32*, i32** %1, align 8
  %3 = getelementptr inbounds i32, i32* %2, i32 %i
  %4 = getelementptr inbounds { i32*, i32* }, { i32*, i32* }* %x, i64 0, i32 0
  store i32* %3, i32** %4, align 8
  %5 = getelementptr inbounds { i32*, i32* }, { i32*, i32* }* %y, i64 0, i32 1
  %6 = load i32*, i32** %5, align 8
  %7 = getelementptr inbounds i32, i32* %6, i32 %i
  %8 = getelementptr inbounds { i32*, i32* }, { i32*, i32* }* %x, i64 0, i32 1
  store i32* %7, i32** %8, align 8
  ret void
}
