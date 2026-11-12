; RUN: opt < %s -passes=gvn -S | FileCheck %s

@last = external global [65 x ptr]

define i32 @NextRootMove(i32 %wtm, i32 %x, i32 %y, i32 %z) {
entry:
        %A = alloca ptr
	%tmp17618 = load ptr, ptr getelementptr ([65 x ptr], ptr @last, i32 0, i32 1), align 4
        store ptr %tmp17618, ptr %A
; CHECK: entry:
; CHECK-NEXT: alloca ptr
; CHECK-NEXT: %tmp17618 = load
; CHECK-NOT: load
; CHECK-NOT: phi
	br label %cond_true116

cond_true116:
   %cmp = icmp eq i32 %x, %y
	br i1 %cmp, label %cond_true128, label %cond_true145

cond_true128:
	%tmp17625 = load ptr, ptr getelementptr ([65 x ptr], ptr @last, i32 0, i32 1), align 4
        store ptr %tmp17625, ptr %A
   %cmp1 = icmp eq i32 %x, %z
	br i1 %cmp1 , label %bb98.backedge, label %return.loopexit

bb98.backedge:
	br label %cond_true116

cond_true145:
	%tmp17631 = load ptr, ptr getelementptr ([65 x ptr], ptr @last, i32 0, i32 1), align 4
        store ptr %tmp17631, ptr %A
	br i1 false, label %bb98.backedge, label %return.loopexit

return.loopexit:
	br label %return

return:
	ret i32 0
}
