; Test merging of blocks with phi nodes.
;
; RUN: opt < %s -simplifycfg -simplifycfg-require-and-preserve-domtree=1 -S > %t
; RUN: not grep N: %t
; RUN: not grep X: %t
; RUN: not grep 'switch i32[^U]+%U' %t
; RUN: not grep "^BB.tomerge" %t
; RUN: grep "^BB.nomerge" %t | count 4
;

; ModuleID = '<stdin>'
declare i1 @foo()

declare i1 @bar(i32)

define i32 @test(i1 %a) {
Q:
        br i1 %a, label %N, label %M
N:              ; preds = %Q
        br label %M
M:              ; preds = %N, %Q
        ; It's ok to merge N and M because the incoming values for W are the
        ; same for both cases...
        %W = phi i32 [ 2, %N ], [ 2, %Q ]               ; <i32> [#uses=1]
        %R = add i32 %W, 1              ; <i32> [#uses=1]
        ret i32 %R
}

; Test merging of blocks with phi nodes where at least one incoming value
; in the successor is undef.
define i8 @testundef(i32 %u) {
R:
  switch i32 %u, label %U [
    i32 0, label %S
    i32 1, label %T
    i32 2, label %T
  ]

S:                                            ; preds = %R
  br label %U

T:                                           ; preds = %R, %R
  br label %U

U:                                        ; preds = %T, %S, %R
  ; We should be able to merge either the S or T block into U by rewriting
  ; R's incoming value with the incoming value of that predecessor since
  ; R's incoming value is undef and both of those predecessors are simple
  ; unconditional branches.
  %val.0 = phi i8 [ undef, %R ], [ 1, %T ], [ 0, %S ]
  ret i8 %val.0
}

; Test merging of blocks with phi nodes where at least one incoming value
; in the successor is undef.
define i8 @testundef2(i32 %u, i32* %A) {
V:
  switch i32 %u, label %U [
    i32 0, label %W
    i32 1, label %X
    i32 2, label %X
    i32 3, label %Z
  ]

W:                                            ; preds = %V
  br label %U

Z:
  store i32 0, i32* %A, align 4
  br label %X

X:                                           ; preds = %V, %V, %Z
  br label %U

U:                                        ; preds = %X, %W, %V
  ; We should be able to merge either the W or X block into U by rewriting
  ; V's incoming value with the incoming value of that predecessor since
  ; V's incoming value is undef and both of those predecessors are simple
  ; unconditional branches. Note that X has predecessors beyond
  ; the direct predecessors of U.
  %val.0 = phi i8 [ undef, %V ], [ 1, %X ], [ 1, %W ]
  ret i8 %val.0
}

define i8 @testmergesome(i32 %u, i32* %A) {
V:
  switch i32 %u, label %Y [
    i32 0, label %W
    i32 1, label %X
    i32 2, label %X
    i32 3, label %Z
  ]

W:                                            ; preds = %V
  store i32 1, i32* %A, align 4
  br label %Y

Z:
  store i32 0, i32* %A, align 4
  br label %X

X:                                           ; preds = %V, %Z
  br label %Y

Y:                                        ; preds = %X, %W, %V
  ; After merging X into Y, we should have 5 predecessors
  ; and thus 5 incoming values to the phi.
  %val.0 = phi i8 [ 1, %V ], [ 1, %X ], [ 2, %W ]
  ret i8 %val.0
}


define i8 @testmergesome2(i32 %u, i32* %A) {
V:
  switch i32 %u, label %W [
    i32 0, label %W
    i32 1, label %Y
    i32 2, label %X
    i32 4, label %Y
  ]

W:                                            ; preds = %V
  store i32 1, i32* %A, align 4
  br label %Y

X:                                           ; preds = %V, %Z
  br label %Y

Y:                                        ; preds = %X, %W, %V
  ; Ensure that we deal with both undef inputs for V when we merge in X.
  %val.0 = phi i8 [ undef, %V ], [ 1, %X ], [ 2, %W ], [ undef, %V ]
  ret i8 %val.0
}

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

; This function can't be merged (for keeping canonical loop structures)
define void @c() {
entry:
	br label %BB.nomerge

BB.nomerge:		; preds = %Common, %entry
	br label %Succ

Succ:		; preds = %Common, %BB.tomerge, %Pre-Exit
        ; This phi has identical values for Common and (through BB) Common,
        ; blocks can't be merged
	%b = phi i32 [ 1, %BB.nomerge ], [ 1, %Common ], [ 2, %Pre-Exit ]
	%conde = call i1 @foo( )		; <i1> [#uses=1]
	br i1 %conde, label %Common, label %Pre-Exit

Common:		; preds = %Succ
	%cond = call i1 @foo( )		; <i1> [#uses=1]
	br i1 %cond, label %BB.nomerge, label %Succ

Pre-Exit:       ; preds = %Succ
        ; This adds a backedge, so the %b phi node gets a third branch and is
        ; not completely trivial
	%cond2 = call i1 @foo( )		; <i1> [#uses=1]
	br i1 %cond2, label %Succ, label %Exit

Exit:		; preds = %Pre-Exit
	ret void
}

; This function can't be merged (for keeping canonical loop structures)
define void @d() {
entry:
	br label %BB.nomerge

BB.nomerge:		; preds = %Common, %entry
        ; This phi has a matching value (0) with below phi (0), so blocks
        ; can be merged.
	%a = phi i32 [ 1, %entry ], [ 0, %Common ]		; <i32> [#uses=1]
	br label %Succ

Succ:		; preds = %Common, %BB.tomerge
	%b = phi i32 [ %a, %BB.nomerge ], [ 0, %Common ]		; <i32> [#uses=0]
	%conde = call i1 @foo( )		; <i1> [#uses=1]
	br i1 %conde, label %Common, label %Exit

Common:		; preds = %Succ
	%cond = call i1 @foo( )		; <i1> [#uses=1]
	br i1 %cond, label %BB.nomerge, label %Succ

Exit:		; preds = %Succ
	ret void
}

; This function can be merged
define void @e() {
entry:
	br label %Succ

Succ:		; preds = %Use, %entry
        ; This phi is used somewhere else than Succ, but this should not prevent
        ; merging this block
	%a = phi i32 [ 1, %entry ], [ 0, %Use ]		; <i32> [#uses=1]
	br label %BB.tomerge

BB.tomerge:		; preds = %Succ
	%conde = call i1 @foo( )		; <i1> [#uses=1]
	br i1 %conde, label %Use, label %Exit

Use:		; preds = %Succ
	%cond = call i1 @bar( i32 %a )		; <i1> [#uses=1]
	br i1 %cond, label %Succ, label %Exit

Exit:		; preds = %Use, %Succ
	ret void
}
