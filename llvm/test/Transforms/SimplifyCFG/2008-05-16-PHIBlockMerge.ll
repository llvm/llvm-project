; RUN: llvm-as < %s | opt -simplifycfg | llvm-dis > %t
; RUN: not grep {^BB.tomerge} %t
; RUN  grep {^BB.nomerge} %t | count 2

; ModuleID = '<stdin>' 
declare i1 @foo()

declare i1 @bar(i32)

; This function can't be merged
define void @a() {
entry:
	br label %BB.nomerge

BB.nomerge:		; preds = %Common, %entry
        ; This phi has a conflicting value (0) with below phi (2), so blocks
        ; can't be merged.
	%a = phi i32 [ 1, %entry ], [ 0, %Common ]		; <i32> [#uses=1]
	br label %Succ

Succ:		; preds = %Common, %BB.nomerge
	%b = phi i32 [ %a, %BB.nomerge ], [ 2, %Common ]		; <i32> [#uses=0]
	%conde = call i1 @foo( )		; <i1> [#uses=1]
	br i1 %conde, label %Common, label %Exit

Common:		; preds = %Succ
	%cond = call i1 @foo( )		; <i1> [#uses=1]
	br i1 %cond, label %BB.nomerge, label %Succ

Exit:		; preds = %Succ
	ret void
}

; This function can't be merged
define void @b() {
entry:
	br label %BB.nomerge

BB.nomerge:		; preds = %Common, %entry
	br label %Succ

Succ:		; preds = %Common, %BB.nomerge
        ; This phi has confliction values for Common and (through BB) Common,
        ; blocks can't be merged
	%b = phi i32 [ 1, %BB.nomerge ], [ 2, %Common ]		; <i32> [#uses=0]
	%conde = call i1 @foo( )		; <i1> [#uses=1]
	br i1 %conde, label %Common, label %Exit

Common:		; preds = %Succ
	%cond = call i1 @foo( )		; <i1> [#uses=1]
	br i1 %cond, label %BB.nomerge, label %Succ

Exit:		; preds = %Succ
	ret void
}

; This function can be merged
define void @c() {
entry:
	br label %BB.tomerge

BB.tomerge:		; preds = %Common, %entry
	br label %Succ

Succ:		; preds = %Common, %BB.tomerge, %Pre-Exit
        ; This phi has identical values for Common and (through BB) Common,
        ; blocks can't be merged
	%b = phi i32 [ 1, %BB.tomerge ], [ 1, %Common ], [ 2, %Pre-Exit ]
	%conde = call i1 @foo( )		; <i1> [#uses=1]
	br i1 %conde, label %Common, label %Pre-Exit

Common:		; preds = %Succ
	%cond = call i1 @foo( )		; <i1> [#uses=1]
	br i1 %cond, label %BB.tomerge, label %Succ

Pre-Exit:       ; preds = %Succ
        ; This adds a backedge, so the %b phi node gets a third branch and is
        ; not completely trivial
	%cond2 = call i1 @foo( )		; <i1> [#uses=1]
	br i1 %cond2, label %Succ, label %Exit
        
Exit:		; preds = %Pre-Exit
	ret void
}

; This function can be merged
define void @d() {
entry:
	br label %BB.tomerge

BB.tomerge:		; preds = %Common, %entry
        ; This phi has a matching value (0) with below phi (0), so blocks
        ; can be merged.
	%a = phi i32 [ 1, %entry ], [ 0, %Common ]		; <i32> [#uses=1]
	br label %Succ

Succ:		; preds = %Common, %BB.tomerge
	%b = phi i32 [ %a, %BB.tomerge ], [ 0, %Common ]		; <i32> [#uses=0]
	%conde = call i1 @foo( )		; <i1> [#uses=1]
	br i1 %conde, label %Common, label %Exit

Common:		; preds = %Succ
	%cond = call i1 @foo( )		; <i1> [#uses=1]
	br i1 %cond, label %BB.tomerge, label %Succ

Exit:		; preds = %Succ
	ret void
}

; This function can be merged
define void @e() {
entry:
	br label %BB.tomerge

BB.tomerge:		; preds = %Use, %entry
        ; This phi is used somewhere else than Succ, but this should not prevent
        ; merging this block
	%a = phi i32 [ 1, %entry ], [ 0, %Use ]		; <i32> [#uses=1]
	br label %Succ

Succ:		; preds = %BB.tomerge
	%conde = call i1 @foo( )		; <i1> [#uses=1]
	br i1 %conde, label %Use, label %Exit

Use:		; preds = %Succ
	%cond = call i1 @bar( i32 %a )		; <i1> [#uses=1]
	br i1 %cond, label %BB.tomerge, label %Exit

Exit:		; preds = %Use, %Succ
	ret void
}
