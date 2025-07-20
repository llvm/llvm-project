; RUN: opt -S -enable-loop-simplifycfg-term-folding=true -passes='require<domtree>,loop(loop-simplifycfg)' -verify-loop-info -verify-dom-info -verify-loop-lcssa < %s | FileCheck %s
 
define void @test() {
; CHECK-LABEL: @test(

  indirectbr ptr null, [label %A, label %C]

A:
  br i1 true, label %B, label %C

B:
  br i1 true, label %A, label %C

C:
  unreachable
}
