; RUN: llvm-as < %s | opt -basicaa -aa-eval -disable-output 2>/dev/null
; Test for a bug in BasicAA which caused a crash when querying equality of P1&P2
define void @test([17 x i16]* %mask_bits) {
	%P1 = getelementptr [17 x i16]* %mask_bits, i64 0, i64 0
	%P2 = getelementptr [17 x i16]* %mask_bits, i64 252645134, i64 0
	ret void
}
