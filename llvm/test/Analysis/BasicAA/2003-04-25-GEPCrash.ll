; RUN: opt < %s -aa-pipeline=basic-aa -passes=aa-eval -disable-output 2>/dev/null
; Test for a bug in BasicAA which caused a crash when querying equality of P1&P2
define void @test(ptr %mask_bits) {
	%P2 = getelementptr [17 x i16], ptr %mask_bits, i64 252645134, i64 0
	ret void
}
