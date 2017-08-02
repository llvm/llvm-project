; RUN: llc < %s -mtriple=i686-- | grep mov | count 1
; PR1874
	
define i32 @test(i32 %a, i32 %b) {
entry:
	%tmp3 = mul i32 %b, %a
	ret i32 %tmp3
}
