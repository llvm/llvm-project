; RUN: llc < %s -march=x86 | grep "mov.*56"
; PR3449

define void @test(<8 x double>* %P, i64* %Q) nounwind {
	%A = load <8 x double>, <8 x double>* %P		; <<8 x double>> [#uses=1]
	%B = bitcast <8 x double> %A to i512		; <i512> [#uses=1]
	%C = lshr i512 %B, 448		; <i512> [#uses=1]
	%D = trunc i512 %C to i64		; <i64> [#uses=1]
	store volatile i64 %D, i64* %Q
	ret void
}
