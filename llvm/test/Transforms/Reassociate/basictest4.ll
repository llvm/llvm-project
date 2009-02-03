; RUN: llvm-as < %s | opt -reassociate -gvn -instcombine | llvm-dis | not grep add

@a = weak global i32 0		; <i32*> [#uses=1]
@b = weak global i32 0		; <i32*> [#uses=1]
@c = weak global i32 0		; <i32*> [#uses=1]
@d = weak global i32 0		; <i32*> [#uses=0]

define i32 @foo() {
	%tmp.0 = load i32* @a		; <i32> [#uses=2]
	%tmp.1 = load i32* @b		; <i32> [#uses=2]
        ; (a+b)
	%tmp.2 = add i32 %tmp.0, %tmp.1		; <i32> [#uses=1]
	%tmp.4 = load i32* @c		; <i32> [#uses=2]
	; (a+b)+c
        %tmp.5 = add i32 %tmp.2, %tmp.4		; <i32> [#uses=1]
	; (a+c)
        %tmp.8 = add i32 %tmp.0, %tmp.4		; <i32> [#uses=1]
	; (a+c)+b
        %tmp.11 = add i32 %tmp.8, %tmp.1		; <i32> [#uses=1]
	; X ^ X = 0
        %RV = xor i32 %tmp.5, %tmp.11		; <i32> [#uses=1]
	ret i32 %RV
}
