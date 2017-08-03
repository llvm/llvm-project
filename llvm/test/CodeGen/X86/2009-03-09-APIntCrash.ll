; RUN: llc < %s -march=x86-64
; PR3763
	%struct.__block_descriptor = type { i64, i64 }

define %struct.__block_descriptor @evUTCTime() nounwind {
entry:
	br i1 false, label %if.then, label %return

if.then:		; preds = %entry
	%srcval18 = load i128, i128* null, align 8		; <i128> [#uses=1]
	%tmp15 = lshr i128 %srcval18, 64		; <i128> [#uses=1]
	%tmp9 = mul i128 %tmp15, 18446744073709551616000		; <i128> [#uses=1]
	br label %return

return:		; preds = %if.then, %entry
	%retval.0 = phi i128 [ %tmp9, %if.then ], [ undef, %entry ]		; <i128> [#uses=0]
	ret %struct.__block_descriptor undef
}

define i128 @test(i128 %arg) nounwind {
	%A = shl i128 1, 92
	%B = sub i128 0, %A
	%C = mul i128 %arg, %B
	ret i128 %C  ;; should codegen to neg(shift)
}
