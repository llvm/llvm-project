; RUN: opt < %s -lcssa -S -verify-loop-info | FileCheck %s
; PR6546

; LCSSA doesn't need to transform uses in blocks not reachable
; from the entry block.

; CHECK: %tmp33 = load i1*, i1** %tmp

define fastcc void @dfs() nounwind {
bb:
  br label %bb44

bb44:
  br i1 undef, label %bb7, label %bb45

bb7:
  %tmp = bitcast i1** undef to i1**
  br label %bb15

bb15:
  br label %bb44

bb32:
  %tmp33 = load i1*, i1** %tmp, align 8
  br label %bb45

bb45:
  unreachable
}
