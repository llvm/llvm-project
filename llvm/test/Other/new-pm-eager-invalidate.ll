; RUN: opt -disable-verify -debug-pass-manager -passes='function(require<no-op-function>)' -disable-output %s 2>&1 | FileCheck %s --check-prefix=NORMAL
; RUN: opt -disable-verify -debug-pass-manager -passes='cgscc(function(require<no-op-function>))' -disable-output %s 2>&1 | FileCheck %s --check-prefix=NORMAL
; RUN: opt -disable-verify -debug-pass-manager -passes='function<eager-inv>(require<no-op-function>)' -disable-output %s 2>&1 | FileCheck %s --check-prefix=EAGER
; RUN: opt -disable-verify -debug-pass-manager -passes='cgscc(function<eager-inv>(require<no-op-function>))' -disable-output %s 2>&1 | FileCheck %s --check-prefix=EAGER

; RUN: opt -disable-verify -debug-pass-manager -passes='default<O2>' -eagerly-invalidate-analyses=0 -disable-output %s 2>&1 | FileCheck %s --check-prefix=PIPELINE
; RUN: opt -disable-verify -debug-pass-manager -passes='default<O2>' -eagerly-invalidate-analyses=1 -disable-output %s 2>&1 | FileCheck %s --check-prefix=PIPELINE-EAGER

; NORMAL-NOT: Invalidating analysis: NoOpFunctionAnalysis
; EAGER: Invalidating analysis: NoOpFunctionAnalysis
; PIPELINE-NOT: Invalidating analysis: DominatorTreeAnalysis
; PIPELINE-EAGER: Invalidating analysis: DominatorTreeAnalysis

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
