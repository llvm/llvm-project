; RUN: opt < %s -constprop -S | grep "ret i13 13"
; PR1816
declare i13 @llvm.cttz.i13(i13, i1)

define i13 @test() {
	%X = call i13 @llvm.cttz.i13(i13 0, i1 true)
	ret i13 %X
}
