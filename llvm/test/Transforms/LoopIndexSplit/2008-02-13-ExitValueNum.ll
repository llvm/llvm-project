; RUN: llvm-as < %s | opt -disable-output -loop-index-split
; PR 2011
	%struct.CLAUSE_HELP = type { i32, i32, i32, i32, i32*, i32, %struct.LIST_NODE*, %struct.LIST_NODE*, i32, i32, %struct.LITERAL_HELP**, i32, i32, i32, i32 }
	%struct.LIST_NODE = type { %struct.LIST_NODE*, i8* }
	%struct.LITERAL_HELP = type { i32, i32, i32, %struct.CLAUSE_HELP*, %struct.term* }
	%struct.anon = type { %struct.LIST_NODE* }
	%struct.st = type { %struct.subst*, %struct.LIST_NODE*, %struct.LIST_NODE*, i16, i16 }
	%struct.subst = type { %struct.subst*, i32, %struct.term* }
	%struct.term = type { i32, %struct.anon, %struct.LIST_NODE*, i32, i32 }

define fastcc %struct.LIST_NODE* @inf_HyperResolvents(%struct.CLAUSE_HELP* %Clause, %struct.subst* %Subst, %struct.LIST_NODE* %Restlits, i32 %GlobalMaxVar, %struct.LIST_NODE* %FoundMap, i32 %StrictlyMaximal, { %struct.st*, [3001 x %struct.term*], [4000 x %struct.term*], i32 }* %Index, i32* %Flags, i32* %Precedence) nounwind  {
entry:
	br i1 false, label %bb960, label %bb885

bb885:		; preds = %entry
	ret %struct.LIST_NODE* null

bb960:		; preds = %entry
	br i1 false, label %bb1097, label %bb1005.preheader

bb1005.preheader:		; preds = %bb960
	ret %struct.LIST_NODE* null

bb1097:		; preds = %bb960
	br i1 false, label %bb1269.preheader, label %bb1141.preheader

bb1141.preheader:		; preds = %bb1097
	ret %struct.LIST_NODE* null

bb1269.preheader:		; preds = %bb1097
	br i1 false, label %bb1318, label %bb1281

bb1281:		; preds = %bb1269.preheader
	ret %struct.LIST_NODE* null

bb1318:		; preds = %bb1269.preheader
	br i1 false, label %bb1459, label %bb.nph52

bb.nph52:		; preds = %bb1318
	ret %struct.LIST_NODE* null

bb1459:		; preds = %bb1318
	br i1 false, label %bb1553, label %bb.nph62

bb.nph62:		; preds = %bb1459
	ret %struct.LIST_NODE* null

bb1553:		; preds = %bb1669, %bb1459
	%j295.0.reg2mem.0 = phi i32 [ %storemerge110, %bb1669 ], [ 0, %bb1459 ]		; <i32> [#uses=2]
	%tmp1629 = icmp sgt i32 %j295.0.reg2mem.0, 0		; <i1> [#uses=1]
	br i1 %tmp1629, label %bb1649, label %bb1632

bb1632:		; preds = %bb1553
	br label %bb1669

bb1649:		; preds = %bb1553
	br label %bb1669

bb1669:		; preds = %bb1649, %bb1632
	%storemerge110 = add i32 %j295.0.reg2mem.0, 1		; <i32> [#uses=2]
	%tmp1672 = icmp sgt i32 %storemerge110, 0		; <i1> [#uses=1]
	br i1 %tmp1672, label %bb1678, label %bb1553

bb1678:		; preds = %bb1669
	ret %struct.LIST_NODE* null
}

