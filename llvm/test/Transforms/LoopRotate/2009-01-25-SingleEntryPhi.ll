; RUN: opt < %s -loop-rotate -verify-dom-info -verify-loop-info -disable-output
; PR3408
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"
target triple = "x86_64-unknown-linux-gnu"
	%struct.Cls = type { i32, i8, [2 x %struct.Cls*], [2 x %struct.Lit*] }
	%struct.Lit = type { i8 }

define void @picosat_main_bb13.i.i71.outer_bb132.i.i.i.outer(%struct.Cls**, %struct.Cls**, i32 %collect.i.i.i.1.lcssa, i32 %lcollect.i.i.i.2.lcssa, %struct.Cls*** %rhead.tmp.0236.out, i32* %collect.i.i.i.2.out, i32* %lcollect.i.i.i.3.ph.ph.ph.out) nounwind {
newFuncRoot:
	br label %codeRepl

bb133.i.i.i.exitStub:		; preds = %codeRepl
	ret void

bb130.i.i.i:		; preds = %codeRepl
	%rhead.tmp.0236.lcssa82 = phi %struct.Cls** [ null, %codeRepl ]		; <%struct.Cls**> [#uses=0]
	br label %codeRepl

codeRepl:		; preds = %bb130.i.i.i, %newFuncRoot
	br i1 false, label %bb130.i.i.i, label %bb133.i.i.i.exitStub
}
