; RUN: llvm-as < %s | opt -predsimplify -verify
target datalayout = "e-p:32:32"
target triple = "i686-pc-linux-gnu"
deplibs = [ "c", "crtend" ]
	%struct.anon = type { %struct.set_family*, %struct.set_family*, %struct.set_family*, i8*, i32, i32*, %struct.pair_struct*, i8**, %struct.symbolic_t*, %struct.symbolic_t* }
	%struct.pair_struct = type { i32, i32*, i32* }
	%struct.set_family = type { i32, i32, i32, i32, i32, i32*, %struct.set_family* }
	%struct.symbolic_label_t = type { i8*, %struct.symbolic_label_t* }
	%struct.symbolic_list_t = type { i32, i32, %struct.symbolic_list_t* }
	%struct.symbolic_t = type { %struct.symbolic_list_t*, i32, %struct.symbolic_label_t*, i32, %struct.symbolic_t* }

define void @find_pairing_cost(i32 %strategy) {
entry:
	br i1 false, label %cond_true299, label %bb314
bb94:		; preds = %cond_true299
	switch i32 %strategy, label %bb246 [
		 i32 0, label %bb196
		 i32 1, label %bb159
	]
cond_next113:		; preds = %cond_true299
	switch i32 %strategy, label %bb246 [
		 i32 0, label %bb196
		 i32 1, label %bb159
	]
bb159:		; preds = %cond_next113, %bb94
	ret void
bb196:		; preds = %cond_next113, %bb94
	%Rsave.0.3 = phi %struct.set_family* [ null, %bb94 ], [ null, %cond_next113 ]		; <%struct.set_family*> [#uses=0]
	ret void
bb246:		; preds = %cond_next113, %bb94
	br label %bb314
cond_true299:		; preds = %entry
	%tmp55 = icmp sgt i32 %strategy, 0		; <i1> [#uses=1]
	br i1 %tmp55, label %bb94, label %cond_next113
bb314:		; preds = %bb246, %entry
	ret void
}
