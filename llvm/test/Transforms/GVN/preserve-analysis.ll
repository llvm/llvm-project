; RUN: opt < %s -debug-pass=Structure -indvars -gvn -indvars -enable-new-pm=0 2>&1 -S | FileCheck --check-prefix=CHECK --check-prefix=IR %s
; RUN: opt < %s -debug-pass-manager -passes='require<domtree>,loop(loop-simplifycfg),gvn,loop(indvars)' 2>&1 -S | FileCheck --check-prefix=NEW-PM --check-prefix=IR %s

; Check CFG-only analysis are preserved by SCCP by running it between 2
; loop-vectorize runs.

; CHECK: Dominator Tree Construction
; CHECK: Natural Loop Information
; CHECK: Canonicalize natural loops
; CHECK: LCSSA Verifier
; CHECK: Loop-Closed SSA Form Pass
; CHECK: Global Value Numbering
; CHECK-NOT: Dominator Tree Construction
; CHECK-NOT: Natural Loop Information
; CHECK: Canonicalize natural loops

; NEW-PM-DAG: Running analysis: LoopAnalysis on test
; NEW-PM-DAG: Running analysis: DominatorTreeAnalysis on test
; NEW-PM: Running pass: GVNPass on test
; NEW-PM-NOT: Running analysis: LoopAnalysis on test
; NEW-PM-NOT: Running analysis: DominatorTreeAnalysis on test

declare i1 @cond()
declare void @dostuff()

define i32 @test() {
; IR-LABEL: define i32 @test()
; IR-LABEL: header:
; IR:         br i1 false, label %then, label %latch
; IR-LABEL: then:
; IR-NEXT:   call void @dostuff()
; IR-NEXT:   br label %latch
entry:
  %res = add i32 1, 10
  br label %header

header:
  %iv = phi i32 [ %res, %entry ], [ 0, %latch ]
  %ic = icmp eq i32 %res, 99
  br i1 %ic, label %then, label %latch

then:
  br label %then.2

then.2:
  call void @dostuff()
  br label %latch


latch:
  %ec = call i1 @cond()
  br i1 %ec, label %exit, label %header

exit:
  ret i32 %iv
}
