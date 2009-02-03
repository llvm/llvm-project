; RUN: llvm-as < %s | llc -march=x86 | grep 8388635
; RUN: llvm-as < %s | llc -march=x86-64 | grep 4294981120
; PR 1325

; ModuleID = 'bugpoint.test.bc'
target datalayout = "E-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64"
target triple = "powerpc-apple-darwin8.8.0"
;target triple = "i686-linux-gnu"
	%struct.FILE = type { i8*, i32, i32, i16, i16, %struct.__sbuf, i32, i8*, i32 (i8*)*, i32 (i8*, i8*, i32)*, i64 (i8*, i64, i32)*, i32 (i8*, i8*, i32)*, %struct.__sbuf, %struct.__sFILEX*, i32, [3 x i8], [1 x i8], %struct.__sbuf, i32, i64 }
	%struct.__sFILEX = type opaque
	%struct.__sbuf = type { i8*, i32 }
@PL_rsfp = external global %struct.FILE*		; <%struct.FILE**> [#uses=1]
@PL_bufend = external global i8*		; <i8**> [#uses=1]
@PL_in_eval = external global i32		; <i32*> [#uses=1]

declare fastcc void @incline(i8*)

define i16 @Perl_skipspace_bb60(i8* %s, i8** %s_addr.4.out) {
newFuncRoot:
	%tmp138.loc = alloca i8*		; <i8**> [#uses=2]
	%s_addr.4.loc = alloca i8*		; <i8**> [#uses=2]
	%tmp274.loc = alloca i8*		; <i8**> [#uses=2]
	br label %bb60

cond_next154.UnifiedReturnBlock_crit_edge.exitStub:		; preds = %codeRepl
	store i8* %s_addr.4.reload, i8** %s_addr.4.out
	ret i16 0

cond_next161.UnifiedReturnBlock_crit_edge.exitStub:		; preds = %codeRepl
	store i8* %s_addr.4.reload, i8** %s_addr.4.out
	ret i16 1

cond_next167.UnifiedReturnBlock_crit_edge.exitStub:		; preds = %codeRepl
	store i8* %s_addr.4.reload, i8** %s_addr.4.out
	ret i16 2

cond_false29.i.cond_true190_crit_edge.exitStub:		; preds = %codeRepl
	store i8* %s_addr.4.reload, i8** %s_addr.4.out
	ret i16 3

cond_next.i.cond_true190_crit_edge.exitStub:		; preds = %codeRepl
	store i8* %s_addr.4.reload, i8** %s_addr.4.out
	ret i16 4

cond_true19.i.cond_true190_crit_edge.exitStub:		; preds = %codeRepl
	store i8* %s_addr.4.reload, i8** %s_addr.4.out
	ret i16 5

bb60:		; preds = %bb60.backedge, %newFuncRoot
	%s_addr.2 = phi i8* [ %s, %newFuncRoot ], [ %s_addr.2.be, %bb60.backedge ]		; <i8*> [#uses=3]
	%tmp61 = load i8** @PL_bufend		; <i8*> [#uses=1]
	%tmp63 = icmp ult i8* %s_addr.2, %tmp61		; <i1> [#uses=1]
	br i1 %tmp63, label %bb60.cond_next67_crit_edge, label %bb60.bb101_crit_edge

bb37:		; preds = %cond_next67.bb37_crit_edge5, %cond_next67.bb37_crit_edge4, %cond_next67.bb37_crit_edge3, %cond_next67.bb37_crit_edge2, %cond_next67.bb37_crit_edge
	%tmp40 = icmp eq i8 %tmp69, 10		; <i1> [#uses=1]
	%tmp43 = getelementptr i8* %s_addr.27.2, i32 1		; <i8*> [#uses=5]
	br i1 %tmp40, label %cond_true45, label %bb37.bb60_crit_edge

cond_true45:		; preds = %bb37
	%tmp46 = volatile load i32* @PL_in_eval		; <i32> [#uses=1]
	%tmp47 = icmp eq i32 %tmp46, 0		; <i1> [#uses=1]
	br i1 %tmp47, label %cond_true45.bb60_crit_edge, label %cond_true50

cond_true50:		; preds = %cond_true45
	%tmp51 = volatile load %struct.FILE** @PL_rsfp		; <%struct.FILE*> [#uses=1]
	%tmp52 = icmp eq %struct.FILE* %tmp51, null		; <i1> [#uses=1]
	br i1 %tmp52, label %cond_true55, label %cond_true50.bb60_crit_edge

cond_true55:		; preds = %cond_true50
	tail call fastcc void @incline( i8* %tmp43 )
	br label %bb60.backedge

cond_next67:		; preds = %Perl_newSV.exit.cond_next67_crit_edge, %cond_true148.cond_next67_crit_edge, %bb60.cond_next67_crit_edge
	%s_addr.27.2 = phi i8* [ %s_addr.2, %bb60.cond_next67_crit_edge ], [ %tmp274.reload, %Perl_newSV.exit.cond_next67_crit_edge ], [ %tmp138.reload, %cond_true148.cond_next67_crit_edge ]		; <i8*> [#uses=3]
	%tmp69 = load i8* %s_addr.27.2		; <i8> [#uses=2]
	switch i8 %tmp69, label %cond_next67.bb101_crit_edge [
		 i8 32, label %cond_next67.bb37_crit_edge
		 i8 9, label %cond_next67.bb37_crit_edge2
		 i8 10, label %cond_next67.bb37_crit_edge3
		 i8 13, label %cond_next67.bb37_crit_edge4
		 i8 12, label %cond_next67.bb37_crit_edge5
	]

codeRepl:		; preds = %bb101.preheader
	%targetBlock = call i16 @Perl_skipspace_bb60_bb101( i8* %s_addr.27.3.ph, i8** %tmp274.loc, i8** %s_addr.4.loc, i8** %tmp138.loc )		; <i16> [#uses=1]
	%tmp274.reload = load i8** %tmp274.loc		; <i8*> [#uses=4]
	%s_addr.4.reload = load i8** %s_addr.4.loc		; <i8*> [#uses=6]
	%tmp138.reload = load i8** %tmp138.loc		; <i8*> [#uses=1]
	switch i16 %targetBlock, label %cond_true19.i.cond_true190_crit_edge.exitStub [
		 i16 0, label %cond_next271.bb60_crit_edge
		 i16 1, label %cond_true290.bb60_crit_edge
		 i16 2, label %cond_true295.bb60_crit_edge
		 i16 3, label %Perl_newSV.exit.cond_next67_crit_edge
		 i16 4, label %cond_true148.cond_next67_crit_edge
		 i16 5, label %cond_next154.UnifiedReturnBlock_crit_edge.exitStub
		 i16 6, label %cond_next161.UnifiedReturnBlock_crit_edge.exitStub
		 i16 7, label %cond_next167.UnifiedReturnBlock_crit_edge.exitStub
		 i16 8, label %cond_false29.i.cond_true190_crit_edge.exitStub
		 i16 9, label %cond_next.i.cond_true190_crit_edge.exitStub
	]

bb37.bb60_crit_edge:		; preds = %bb37
	br label %bb60.backedge

cond_true45.bb60_crit_edge:		; preds = %cond_true45
	br label %bb60.backedge

cond_true50.bb60_crit_edge:		; preds = %cond_true50
	br label %bb60.backedge

bb60.cond_next67_crit_edge:		; preds = %bb60
	br label %cond_next67

bb60.bb101_crit_edge:		; preds = %bb60
	br label %bb101.preheader

cond_next67.bb101_crit_edge:		; preds = %cond_next67
	br label %bb101.preheader

cond_next67.bb37_crit_edge:		; preds = %cond_next67
	br label %bb37

cond_next67.bb37_crit_edge2:		; preds = %cond_next67
	br label %bb37

cond_next67.bb37_crit_edge3:		; preds = %cond_next67
	br label %bb37

cond_next67.bb37_crit_edge4:		; preds = %cond_next67
	br label %bb37

cond_next67.bb37_crit_edge5:		; preds = %cond_next67
	br label %bb37

cond_true148.cond_next67_crit_edge:		; preds = %codeRepl
	br label %cond_next67

cond_next271.bb60_crit_edge:		; preds = %codeRepl
	br label %bb60.backedge

cond_true290.bb60_crit_edge:		; preds = %codeRepl
	br label %bb60.backedge

cond_true295.bb60_crit_edge:		; preds = %codeRepl
	br label %bb60.backedge

Perl_newSV.exit.cond_next67_crit_edge:		; preds = %codeRepl
	br label %cond_next67

bb101.preheader:		; preds = %cond_next67.bb101_crit_edge, %bb60.bb101_crit_edge
	%s_addr.27.3.ph = phi i8* [ %s_addr.27.2, %cond_next67.bb101_crit_edge ], [ %s_addr.2, %bb60.bb101_crit_edge ]		; <i8*> [#uses=1]
	br label %codeRepl

bb60.backedge:		; preds = %cond_true295.bb60_crit_edge, %cond_true290.bb60_crit_edge, %cond_next271.bb60_crit_edge, %cond_true50.bb60_crit_edge, %cond_true45.bb60_crit_edge, %bb37.bb60_crit_edge, %cond_true55
	%s_addr.2.be = phi i8* [ %tmp43, %cond_true55 ], [ %tmp43, %bb37.bb60_crit_edge ], [ %tmp43, %cond_true45.bb60_crit_edge ], [ %tmp43, %cond_true50.bb60_crit_edge ], [ %tmp274.reload, %cond_next271.bb60_crit_edge ], [ %tmp274.reload, %cond_true290.bb60_crit_edge ], [ %tmp274.reload, %cond_true295.bb60_crit_edge ]		; <i8*> [#uses=1]
	br label %bb60
}

declare i16 @Perl_skipspace_bb60_bb101(i8*, i8**, i8**, i8**)
