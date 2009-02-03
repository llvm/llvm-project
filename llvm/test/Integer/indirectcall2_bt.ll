; RUN: llvm-as %s -o - | llvm-dis > %t1.ll
; RUN: llvm-as %t1.ll -o - | llvm-dis > %t2.ll
; RUN: diff %t1.ll %t2.ll


define i63 @"test"(i63 %X)
begin
	ret i63 %X
end

define i63 @"fib"(i63 %n)
begin
  %T = icmp ult i63 %n, 2       ; {i1}:0
  br i1 %T, label %BaseCase, label %RecurseCase

RecurseCase:
  %result = call i63 @test(i63 %n)
  br label %BaseCase

BaseCase:
  %X = phi i63 [1, %0], [2, %RecurseCase]
  ret i63 %X
end

