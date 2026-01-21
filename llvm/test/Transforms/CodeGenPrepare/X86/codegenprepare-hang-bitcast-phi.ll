; RUN: opt -mtriple=x86_64-unknown-linux-gnu -passes="require<profile-summary>,function(codegenprepare)" < %s | FileCheck %s
; Check that CodeGenPrepare does not hang on this input.
; This was caused by an infinite loop between OptimizeNoopCopyExpression
; and optimizePhiType when handling same-type bitcasts.

define void @foo() {
entry:
  %val = load i32, ptr null, align 4
  br i1 false, label %bb1, label %bb3

bb1:
  %c1 = bitcast i32 %val to i32
  br label %bb3

bb3:
  %phi = phi i32 [ %c1, %bb1 ], [ %val, %entry ]
  store i32 %phi, ptr null, align 4
  ret void
}
