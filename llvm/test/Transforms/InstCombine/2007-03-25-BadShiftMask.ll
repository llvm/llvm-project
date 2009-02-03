; PR1271
; RUN: llvm-as < %s | opt -instcombine | llvm-dis | \
; RUN:    grep {icmp eq i32 .tmp.*, 2146435072}
%struct..0anon = type { i32, i32 }
%struct..1anon = type { double }

define i32 @main() {
entry:
	%u = alloca %struct..1anon, align 8		; <%struct..1anon*> [#uses=4]
	%tmp1 = getelementptr %struct..1anon* %u, i32 0, i32 0		; <double*> [#uses=1]
	store double 0x7FF0000000000000, double* %tmp1
	%tmp3 = getelementptr %struct..1anon* %u, i32 0, i32 0		; <double*> [#uses=1]
	%tmp34 = bitcast double* %tmp3 to %struct..0anon*		; <%struct..0anon*> [#uses=1]
	%tmp5 = getelementptr %struct..0anon* %tmp34, i32 0, i32 1		; <i32*> [#uses=1]
	%tmp6 = load i32* %tmp5		; <i32> [#uses=1]
	%tmp7 = shl i32 %tmp6, 1		; <i32> [#uses=1]
	%tmp8 = lshr i32 %tmp7, 21		; <i32> [#uses=1]
	%tmp89 = trunc i32 %tmp8 to i16		; <i16> [#uses=1]
	icmp ne i16 %tmp89, 2047		; <i1>:0 [#uses=1]
	zext i1 %0 to i8		; <i8>:1 [#uses=1]
	icmp ne i8 %1, 0		; <i1>:2 [#uses=1]
	br i1 %2, label %cond_true, label %cond_false

cond_true:		; preds = %entry
	ret i32 0

cond_false:		; preds = %entry
        ret i32 1
}
