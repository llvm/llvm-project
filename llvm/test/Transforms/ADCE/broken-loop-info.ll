; RUN: opt -passes='loop(licm),adce,loop(licm)' -S < %s | FileCheck %s
;
; XFAIL: *
; REQUIRES: asserts
;
; This test demonstrates a bug in ADCE's work with loop info. It does some
; changes that make loop's block unreachable, but never bothers to update
; loop info accordingly.

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128-ni:1-p2:32:8:8:32-ni:2"
target triple = "x86_64-unknown-linux-gnu"

define void @test() {
; CHECK-LABEL: test

bb:
  br label %bb2

bb1:                                              ; preds = %bb4
  ret void

bb2:                                              ; preds = %bb4, %bb
  br i1 undef, label %bb4, label %bb3

bb3:                                              ; preds = %bb2
  br label %bb4

bb4:                                              ; preds = %bb3, %bb2
  br i1 undef, label %bb1, label %bb2
}
