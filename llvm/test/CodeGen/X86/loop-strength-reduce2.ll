; RUN: llc < %s -mtriple=i686-apple-darwin -relocation-model=pic | FileCheck %s
;
; Make sure the PIC label flags2-"L1$pb" is not moved up to the preheader.
; CHECK: mov{{.}} {{.*}}$pb

@flags2 = internal global [8193 x i8] zeroinitializer, align 32		; <[8193 x i8]*> [#uses=1]

define void @test(i32 %i) nounwind {
entry:
	%k_addr.012 = shl i32 %i, 1		; <i32> [#uses=1]
	%tmp14 = icmp sgt i32 %k_addr.012, 8192		; <i1> [#uses=1]
	br i1 %tmp14, label %return, label %bb

bb:		; preds = %bb, %entry
	%indvar = phi i32 [ 0, %entry ], [ %indvar.next, %bb ]		; <i32> [#uses=2]
	%tmp. = shl i32 %i, 1		; <i32> [#uses=1]
	%tmp.15 = mul i32 %indvar, %i		; <i32> [#uses=1]
	%tmp.16 = add i32 %tmp.15, %tmp.		; <i32> [#uses=2]
	%k_addr.0.0 = bitcast i32 %tmp.16 to i32		; <i32> [#uses=1]
	%gep.upgrd.1 = zext i32 %tmp.16 to i64		; <i64> [#uses=1]
	%tmp = getelementptr [8193 x i8], [8193 x i8]* @flags2, i32 0, i64 %gep.upgrd.1		; <i8*> [#uses=1]
	store i8 0, i8* %tmp
	%k_addr.0 = add i32 %k_addr.0.0, %i		; <i32> [#uses=1]
	%tmp.upgrd.2 = icmp sgt i32 %k_addr.0, 8192		; <i1> [#uses=1]
	%indvar.next = add i32 %indvar, 1		; <i32> [#uses=1]
	br i1 %tmp.upgrd.2, label %return, label %bb

return:		; preds = %bb, %entry
	ret void
}
