; ModuleID = 'PR1657.bc'
; Do not promote getelementptr because it may exposes load from a null pointer 
; and store from a null pointer  which are covered by 
; icmp eq %struct.decision* null, null condition.
; RUN: llvm-as < %s | opt -licm | llvm-dis | not grep promoted
	%struct.decision = type { i8, %struct.decision* }

define i32 @main() {
entry:
	br label %blah.i

blah.i:		; preds = %cond_true.i, %entry
	%tmp3.i = icmp eq %struct.decision* null, null		; <i1> [#uses=1]
	br i1 %tmp3.i, label %clear_modes.exit, label %cond_true.i

cond_true.i:		; preds = %blah.i
	%tmp1.i = getelementptr %struct.decision* null, i32 0, i32 0		; <i8*> [#uses=1]
	store i8 0, i8* %tmp1.i
	br label %blah.i

clear_modes.exit:		; preds = %blah.i
	call void @exit( i32 0 )
	unreachable
}

define i32 @f(i8* %ptr) {
entry:
        br label %loop.head

loop.head:              ; preds = %cond.true, %entry
        %x = phi i8* [ %ptr, %entry ], [ %ptr.i, %cond.true ]           ; <i8*> [#uses=1]
        %tmp3.i = icmp ne i8* %ptr, %x          ; <i1> [#uses=1]
        br i1 %tmp3.i, label %cond.true, label %exit

cond.true:              ; preds = %loop.head
        %ptr.i = getelementptr i8* %ptr, i32 0          ; <i8*> [#uses=2]
        store i8 0, i8* %ptr.i
        br label %loop.head

exit:           ; preds = %loop.head
        ret i32 0
}

define i32 @f2(i8* %p, i8* %q) {
entry:
        br label %loop.head

loop.head:              ; preds = %cond.true, %entry
        %tmp3.i = icmp eq i8* null, %q            ; <i1> [#uses=1]
        br i1 %tmp3.i, label %exit, label %cond.true

cond.true:              ; preds = %loop.head
        %ptr.i = getelementptr i8* %p, i32 0          ; <i8*> [#uses=2]
        store i8 0, i8* %ptr.i
        br label %loop.head

exit:           ; preds = %loop.head
        ret i32 0
}

declare void @exit(i32)
