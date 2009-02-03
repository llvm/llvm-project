; RUN: llvm-as < %s | llc -march=ppc32 | \
; RUN:   grep {b LBB.*} | count 4

target datalayout = "E-p:32:32"
target triple = "powerpc-apple-darwin8.7.0"

define void @foo(i32 %W, i32 %X, i32 %Y, i32 %Z) {
entry:
	%tmp1 = and i32 %W, 1		; <i32> [#uses=1]
	%tmp1.upgrd.1 = icmp eq i32 %tmp1, 0		; <i1> [#uses=1]
	br i1 %tmp1.upgrd.1, label %cond_false, label %bb5
bb:		; preds = %bb5, %bb
	%indvar77 = phi i32 [ %indvar.next78, %bb ], [ 0, %bb5 ]		; <i32> [#uses=1]
	%tmp2 = tail call i32 (...)* @bar( )		; <i32> [#uses=0]
	%indvar.next78 = add i32 %indvar77, 1		; <i32> [#uses=2]
	%exitcond79 = icmp eq i32 %indvar.next78, %X		; <i1> [#uses=1]
	br i1 %exitcond79, label %cond_next48, label %bb
bb5:		; preds = %entry
	%tmp = icmp eq i32 %X, 0		; <i1> [#uses=1]
	br i1 %tmp, label %cond_next48, label %bb
cond_false:		; preds = %entry
	%tmp10 = and i32 %W, 2		; <i32> [#uses=1]
	%tmp10.upgrd.2 = icmp eq i32 %tmp10, 0		; <i1> [#uses=1]
	br i1 %tmp10.upgrd.2, label %cond_false20, label %bb16
bb12:		; preds = %bb16, %bb12
	%indvar72 = phi i32 [ %indvar.next73, %bb12 ], [ 0, %bb16 ]		; <i32> [#uses=1]
	%tmp13 = tail call i32 (...)* @bar( )		; <i32> [#uses=0]
	%indvar.next73 = add i32 %indvar72, 1		; <i32> [#uses=2]
	%exitcond74 = icmp eq i32 %indvar.next73, %Y		; <i1> [#uses=1]
	br i1 %exitcond74, label %cond_next48, label %bb12
bb16:		; preds = %cond_false
	%tmp18 = icmp eq i32 %Y, 0		; <i1> [#uses=1]
	br i1 %tmp18, label %cond_next48, label %bb12
cond_false20:		; preds = %cond_false
	%tmp23 = and i32 %W, 4		; <i32> [#uses=1]
	%tmp23.upgrd.3 = icmp eq i32 %tmp23, 0		; <i1> [#uses=1]
	br i1 %tmp23.upgrd.3, label %cond_false33, label %bb29
bb25:		; preds = %bb29, %bb25
	%indvar67 = phi i32 [ %indvar.next68, %bb25 ], [ 0, %bb29 ]		; <i32> [#uses=1]
	%tmp26 = tail call i32 (...)* @bar( )		; <i32> [#uses=0]
	%indvar.next68 = add i32 %indvar67, 1		; <i32> [#uses=2]
	%exitcond69 = icmp eq i32 %indvar.next68, %Z		; <i1> [#uses=1]
	br i1 %exitcond69, label %cond_next48, label %bb25
bb29:		; preds = %cond_false20
	%tmp31 = icmp eq i32 %Z, 0		; <i1> [#uses=1]
	br i1 %tmp31, label %cond_next48, label %bb25
cond_false33:		; preds = %cond_false20
	%tmp36 = and i32 %W, 8		; <i32> [#uses=1]
	%tmp36.upgrd.4 = icmp eq i32 %tmp36, 0		; <i1> [#uses=1]
	br i1 %tmp36.upgrd.4, label %cond_next48, label %bb42
bb38:		; preds = %bb42
	%tmp39 = tail call i32 (...)* @bar( )		; <i32> [#uses=0]
	%indvar.next = add i32 %indvar, 1		; <i32> [#uses=1]
	br label %bb42
bb42:		; preds = %bb38, %cond_false33
	%indvar = phi i32 [ %indvar.next, %bb38 ], [ 0, %cond_false33 ]		; <i32> [#uses=4]
	%W_addr.0 = sub i32 %W, %indvar		; <i32> [#uses=1]
	%exitcond = icmp eq i32 %indvar, %W		; <i1> [#uses=1]
	br i1 %exitcond, label %cond_next48, label %bb38
cond_next48:		; preds = %bb42, %cond_false33, %bb29, %bb25, %bb16, %bb12, %bb5, %bb
	%W_addr.1 = phi i32 [ %W, %bb5 ], [ %W, %bb16 ], [ %W, %bb29 ], [ %W, %cond_false33 ], [ %W_addr.0, %bb42 ], [ %W, %bb25 ], [ %W, %bb12 ], [ %W, %bb ]		; <i32> [#uses=1]
	%tmp50 = icmp eq i32 %W_addr.1, 0		; <i1> [#uses=1]
	br i1 %tmp50, label %UnifiedReturnBlock, label %cond_true51
cond_true51:		; preds = %cond_next48
	%tmp52 = tail call i32 (...)* @bar( )		; <i32> [#uses=0]
	ret void
UnifiedReturnBlock:		; preds = %cond_next48
	ret void
}

declare i32 @bar(...)
