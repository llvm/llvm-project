; XFAIL: *
; REQUIRES: asserts
; RUN: opt -S -enable-loop-simplifycfg-term-folding=true -loop-simplifycfg -debug-only=loop-simplifycfg -verify-loop-info -verify-dom-info -verify-loop-lcssa 2>&1 < %s | FileCheck %s
; RUN: opt -S -enable-loop-simplifycfg-term-folding=true -passes='require<domtree>,loop(simplify-cfg)' -debug-only=loop-simplifycfg -verify-loop-info -verify-dom-info -verify-loop-lcssa 2>&1 < %s | FileCheck %s
; RUN: opt -S -enable-loop-simplifycfg-term-folding=true -loop-simplifycfg -enable-mssa-loop-dependency=true -verify-memoryssa -debug-only=loop-simplifycfg -verify-loop-info -verify-dom-info -verify-loop-lcssa 2>&1 < %s | FileCheck %s

target triple = "x86_64-unknown-linux-gnu"

define void @test() {

; CHECK-LABEL: @test(

  br label %bb1

bb1:                                              ; preds = %bb4, %0
  br label %bb2

bb2:                                              ; preds = %bb6, %bb1
  br label %bb3

bb3:                                              ; preds = %bb8, %bb3, %bb2
  br i1 false, label %bb4, label %bb3

bb4:                                              ; preds = %bb8, %bb3
  br i1 undef, label %bb1, label %bb6

bb6:                                              ; preds = %bb4
  br i1 undef, label %bb2, label %bb8

bb8:                                              ; preds = %bb6
  br i1 true, label %bb4, label %bb3
}
