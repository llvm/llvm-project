; RUN: llc < %s -mtriple=i386-apple-darwin | grep mov | count 4

define i16 @test(i16* %tmp179) nounwind  {
	%tmp180 = load i16* %tmp179, align 2		; <i16> [#uses=2]
	%tmp184 = and i16 %tmp180, -1024		; <i16> [#uses=1]
	%tmp186 = icmp eq i16 %tmp184, -32768		; <i1> [#uses=1]
	br i1 %tmp186, label %bb189, label %bb288

bb189:		; preds = %0
	ret i16 %tmp180

bb288:		; preds = %0
	ret i16 32
}
