; RUN: llc < %s -march=x86 -stats  2>&1 | \
; RUN:   grep asm-printer | grep 13

define void @_ZN9__gnu_cxx9hashtableISt4pairIKPKciES3_NS_4hashIS3_EESt10_Select1stIS5_E5eqstrSaIiEE14find_or_insertERKS5__cond_true456.i(i8* %tmp435.i, i32* %tmp449.i.out) nounwind {
newFuncRoot:
	br label %cond_true456.i
bb459.i.exitStub:		; preds = %cond_true456.i
	store i32 %tmp449.i, i32* %tmp449.i.out
	ret void
cond_true456.i:		; preds = %cond_true456.i, %newFuncRoot
	%__s441.2.4.i = phi i8* [ %tmp451.i.upgrd.1, %cond_true456.i ], [ %tmp435.i, %newFuncRoot ]		; <i8*> [#uses=2]
	%__h.2.4.i = phi i32 [ %tmp449.i, %cond_true456.i ], [ 0, %newFuncRoot ]	; <i32> [#uses=1]
	%tmp446.i = mul i32 %__h.2.4.i, 5		; <i32> [#uses=1]
	%tmp.i = load i8* %__s441.2.4.i		; <i8> [#uses=1]
	%tmp448.i = sext i8 %tmp.i to i32		; <i32> [#uses=1]
	%tmp449.i = add i32 %tmp448.i, %tmp446.i		; <i32> [#uses=2]
	%tmp450.i = ptrtoint i8* %__s441.2.4.i to i32		; <i32> [#uses=1]
	%tmp451.i = add i32 %tmp450.i, 1		; <i32> [#uses=1]
	%tmp451.i.upgrd.1 = inttoptr i32 %tmp451.i to i8*		; <i8*> [#uses=2]
	%tmp45435.i = load i8* %tmp451.i.upgrd.1		; <i8> [#uses=1]
	%tmp45536.i = icmp eq i8 %tmp45435.i, 0		; <i1> [#uses=1]
	br i1 %tmp45536.i, label %bb459.i.exitStub, label %cond_true456.i
}

