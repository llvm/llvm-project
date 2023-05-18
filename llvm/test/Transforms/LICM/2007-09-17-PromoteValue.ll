; ModuleID = 'PR1657.bc'
; Do not promote getelementptr because it may exposes load from a null pointer
; and store from a null pointer  which are covered by
; icmp eq ptr null, null condition.
; RUN: opt < %s -passes=licm -S | not grep promoted
	%struct.decision = type { i8, ptr }

define i32 @main() {
entry:
	br label %blah.i

blah.i:		; preds = %cond_true.i, %entry
	%tmp3.i = icmp eq ptr null, null		; <i1> [#uses=1]
	br i1 %tmp3.i, label %clear_modes.exit, label %cond_true.i

cond_true.i:		; preds = %blah.i
	%tmp1.i = getelementptr %struct.decision, ptr null, i32 0, i32 0		; <ptr> [#uses=1]
	store i8 0, ptr %tmp1.i
	br label %blah.i

clear_modes.exit:		; preds = %blah.i
	call void @exit( i32 0 )
	unreachable
}

define i32 @f(ptr %ptr) {
entry:
        br label %loop.head

loop.head:              ; preds = %cond.true, %entry
        %x = phi ptr [ %ptr, %entry ], [ %ptr.i, %cond.true ]           ; <ptr> [#uses=1]
        %tmp3.i = icmp ne ptr %ptr, %x          ; <i1> [#uses=1]
        br i1 %tmp3.i, label %cond.true, label %exit

cond.true:              ; preds = %loop.head
        %ptr.i = getelementptr i8, ptr %ptr, i32 0          ; <ptr> [#uses=2]
        store i8 0, ptr %ptr.i
        br label %loop.head

exit:           ; preds = %loop.head
        ret i32 0
}

define i32 @f2(ptr %p, ptr %q) {
entry:
        br label %loop.head

loop.head:              ; preds = %cond.true, %entry
        %tmp3.i = icmp eq ptr null, %q            ; <i1> [#uses=1]
        br i1 %tmp3.i, label %exit, label %cond.true

cond.true:              ; preds = %loop.head
        %ptr.i = getelementptr i8, ptr %p, i32 0          ; <ptr> [#uses=2]
        store i8 0, ptr %ptr.i
        br label %loop.head

exit:           ; preds = %loop.head
        ret i32 0
}

declare void @exit(i32)
