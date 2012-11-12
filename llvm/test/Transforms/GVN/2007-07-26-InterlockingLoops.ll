; RUN: opt < %s -basicaa -gvn -S | FileCheck %s

@last = external global [65 x i32*]

define i32 @NextRootMove(i32 %wtm) {
entry:
        %A = alloca i32*
	%tmp17618 = load i32** getelementptr ([65 x i32*]* @last, i32 0, i32 1), align 4
        store i32* %tmp17618, i32** %A
; CHECK: entry:
; CHECK-NEXT: alloca i32
; CHECK-NEXT: %tmp17618 = load
; CHECK-NOT: load
; CHECK-NOT: phi
	br label %cond_true116

cond_true116:
	br i1 false, label %cond_true128, label %cond_true145

cond_true128:
	%tmp17625 = load i32** getelementptr ([65 x i32*]* @last, i32 0, i32 1), align 4
        store i32* %tmp17625, i32** %A
	br i1 false, label %bb98.backedge, label %return.loopexit

bb98.backedge:
	br label %cond_true116

cond_true145:
	%tmp17631 = load i32** getelementptr ([65 x i32*]* @last, i32 0, i32 1), align 4
        store i32* %tmp17631, i32** %A
	br i1 false, label %bb98.backedge, label %return.loopexit

return.loopexit:
	br label %return

return:
	ret i32 0
}
