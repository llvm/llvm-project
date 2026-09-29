; RUN: opt -mtriple=hexagon -disable-verify -debug-pass-manager \
; RUN:     -disable-output -passes='default<O1>' -S %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=NPM
; RUN: opt -mtriple=hexagon -disable-verify -debug-pass-manager \
; RUN:     -disable-output -passes='default<O2>' -S %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=NPM
; RUN: opt -mtriple=hexagon -disable-verify -debug-pass-manager \
; RUN:     -disable-output -passes='default<O3>' -S %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=NPM

; Test TargetMachine::registerPassBuilderCallbacks
; NPM: Running pass: HexagonVectorLoopCarriedReusePass

declare void @bar() local_unnamed_addr

define void @foo(i32 %n) local_unnamed_addr {
entry:
  br label %loop
loop:
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %loop ]
  %iv.next = add i32 %iv, 1
  tail call void @bar()
  %cmp = icmp eq i32 %iv, %n
  br i1 %cmp, label %exit, label %loop
exit:
  ret void
}
