; RUN: opt < %s -passes=sccp -S | grep undef | count 1
; PR1938

define i32 @main() {
entry:
	br label %bb

bb:
	%indvar = phi i32 [ 0, %entry ], [ %k, %bb.backedge ]
	%k = add i32 %indvar, 1
	br i1 undef, label %cond_true, label %cond_false

cond_true:
	%tmp97 = icmp slt i32 %k, 10
	br i1 %tmp97, label %bb.backedge, label %bb12

bb.backedge:
	br label %bb

cond_false:
	%tmp9 = icmp slt i32 %k, 10
	br i1 %tmp9, label %bb.backedge, label %bb12

bb12:
	%tmp14 = icmp eq i32 %k, 10
	br i1 %tmp14, label %cond_next18, label %cond_true17

cond_true17:
	tail call void @abort( )
	unreachable

cond_next18:
	ret i32 0
}

declare void @abort()
