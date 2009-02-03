; RUN: llvm-as < %s | opt -constprop | llvm-dis | grep {ret i13 13}
; PR1816
declare i13 @llvm.cttz.i13(i13)

define i13 @test() {
	%X = call i13 @llvm.cttz.i13(i13 0)
	ret i13 %X
}
