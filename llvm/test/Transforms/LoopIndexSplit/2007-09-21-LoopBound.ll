; PR1692
; RUN: llvm-as < %s | opt -loop-index-split -disable-output 
	%struct.CLAUSE_HELP = type { i32, i32, i32, i32, i32*, i32, %struct.LIST_NODE*, %struct.LIST_NODE*, i32, i32, %struct.LITERAL_HELP**, i32, i32, i32, i32 }
	%struct.LIST_NODE = type { %struct.LIST_NODE*, i8* }
	%struct.LITERAL_HELP = type { i32, i32, i32, %struct.CLAUSE_HELP*, %struct.term* }
	%struct.anon = type { %struct.LIST_NODE* }
	%struct.st = type { %struct.subst*, %struct.LIST_NODE*, %struct.LIST_NODE*, i16, i16 }
	%struct.subst = type { %struct.subst*, i32, %struct.term* }
	%struct.term = type { i32, %struct.anon, %struct.LIST_NODE*, i32, i32 }

define %struct.LIST_NODE* @inf_HyperResolvents(%struct.CLAUSE_HELP* %Clause, %struct.subst* %Subst, %struct.LIST_NODE* %Restlits, i32 %GlobalMaxVar, %struct.LIST_NODE* %FoundMap, i32 %StrictlyMaximal, { %struct.st*, [3001 x %struct.term*], [4000 x %struct.term*], i32 }* %Index, i32* %Flags, i32* %Precedence) {
entry:
	br i1 false, label %cond_next44, label %bb37

bb37:		; preds = %entry
	ret %struct.LIST_NODE* null

cond_next44:		; preds = %entry
	br i1 false, label %bb29.i, label %bb.i31

bb.i31:		; preds = %cond_next44
	ret %struct.LIST_NODE* null

bb29.i:		; preds = %cond_next44
	br i1 false, label %cond_next89.i, label %bb34.i

bb34.i:		; preds = %bb29.i
	ret %struct.LIST_NODE* null

cond_next89.i:		; preds = %bb29.i
	br i1 false, label %clause_LiteralGetIndex.exit70.i, label %bb.i59.i

bb.i59.i:		; preds = %cond_next89.i
	ret %struct.LIST_NODE* null

clause_LiteralGetIndex.exit70.i:		; preds = %cond_next89.i
	br label %bb3.i.i

bb3.i.i:		; preds = %bb3.i.i, %clause_LiteralGetIndex.exit70.i
	br i1 false, label %bb40.i.i, label %bb3.i.i

subst_Apply.exit.i.i:		; preds = %bb40.i.i
	%tmp21.i.i = icmp sgt i32 %j.0.i.i, 0		; <i1> [#uses=1]
	br i1 %tmp21.i.i, label %cond_false.i47.i, label %cond_true24.i.i

cond_true24.i.i:		; preds = %subst_Apply.exit.i.i
	br label %cond_next37.i.i

cond_false.i47.i:		; preds = %subst_Apply.exit.i.i
	br label %cond_next37.i.i

cond_next37.i.i:		; preds = %cond_false.i47.i, %cond_true24.i.i
	%tmp39.i.i = add i32 %j.0.i.i, 1		; <i32> [#uses=1]
	br label %bb40.i.i

bb40.i.i:		; preds = %cond_next37.i.i, %bb3.i.i
	%j.0.i.i = phi i32 [ %tmp39.i.i, %cond_next37.i.i ], [ 0, %bb3.i.i ]		; <i32> [#uses=3]
	%tmp43.i.i = icmp sgt i32 %j.0.i.i, 0		; <i1> [#uses=1]
	br i1 %tmp43.i.i, label %inf_CopyHyperElectron.exit.i, label %subst_Apply.exit.i.i

inf_CopyHyperElectron.exit.i:		; preds = %bb40.i.i
	ret %struct.LIST_NODE* null
}
