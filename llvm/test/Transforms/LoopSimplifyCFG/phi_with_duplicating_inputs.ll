; This is currently failing because of bug in LoopSimplifyCFG. It does not update
; duplicating Phi inputs properly.
; XFAIL: *
; RUN: opt -S -enable-loop-simplifycfg-term-folding=true -loop-simplifycfg -debug-only=loop-simplifycfg -verify-loop-info -verify-dom-info -verify-loop-lcssa 2>&1 < %s | FileCheck %s
; RUN: opt -S -enable-loop-simplifycfg-term-folding=true -passes='require<domtree>,loop(simplify-cfg)' -debug-only=loop-simplifycfg -verify-loop-info -verify-dom-info -verify-loop-lcssa 2>&1 < %s | FileCheck %s
; RUN: opt -S -enable-loop-simplifycfg-term-folding=true -loop-simplifycfg -enable-mssa-loop-dependency=true -verify-memoryssa -debug-only=loop-simplifycfg -verify-loop-info -verify-dom-info -verify-loop-lcssa 2>&1 < %s | FileCheck %s

target datalayout = "P40"

@a = external global i16, align 1

; CHECK-LABEL: @f1(

define void @f1(i1 %cond) {
entry:
  br label %for.cond

for.cond:
  br i1 %cond, label %if.then, label %for.inc

if.then:
  %0 = load i16, i16* @a, align 1
  %tobool = icmp ne i16 %0, 0
  br i1 %tobool, label %for.inc, label %for.inc

for.inc:
  %c.1 = phi i16 [ 2, %if.then ], [ 2, %if.then ], [ 1, %for.cond ]
  br label %for.cond
}
