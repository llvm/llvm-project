; RUN: opt < %s -loop-reduce | llvm-dis
; PR3571

target triple = "i386-pc-mingw32"
define void @_ZNK18qdesigner_internal10TreeWidget12drawBranchesEP8QPainterRK5QRectRK11QModelIndex() nounwind {
entry:
	br label %_ZNK11QModelIndex7isValidEv.exit.i

bb.i:		; preds = %_ZNK11QModelIndex7isValidEv.exit.i
	%indvar.next = add i32 %result.0.i, 1		; <i32> [#uses=1]
	br label %_ZNK11QModelIndex7isValidEv.exit.i

_ZNK11QModelIndex7isValidEv.exit.i:		; preds = %bb.i, %entry
	%result.0.i = phi i32 [ 0, %entry ], [ %indvar.next, %bb.i ]		; <i32> [#uses=2]
	%0 = load i32*, i32** null, align 4		; <%struct.QAbstractItemDelegate*> [#uses=0]
	br i1 false, label %_ZN18qdesigner_internalL5levelEP18QAbstractItemModelRK11QModelIndex.exit, label %bb.i

_ZN18qdesigner_internalL5levelEP18QAbstractItemModelRK11QModelIndex.exit:		; preds = %_ZNK11QModelIndex7isValidEv.exit.i
	%1 = call i32 @_ZNK9QTreeView11indentationEv(i32* null) nounwind		; <i32> [#uses=1]
	%2 = mul i32 %1, %result.0.i		; <i32> [#uses=1]
	%3 = add i32 %2, -2		; <i32> [#uses=1]
	%4 = add i32 %3, 0		; <i32> [#uses=1]
	store i32 %4, i32* null, align 8
	unreachable
}

declare i32 @_ZNK9QTreeView11indentationEv(i32*)
