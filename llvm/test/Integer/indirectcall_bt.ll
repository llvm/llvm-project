; RUN: llvm-as %s -o - | llvm-dis > %t1.ll
; RUN: llvm-as %t1.ll -o - | llvm-dis > %t2.ll
; RUN: diff %t1.ll %t2.ll


declare i32 @"atoi"(i8 *)

define i63 @"fib"(i63 %n)
begin
  icmp ult i63 %n, 2       ; {i1}:1
  br i1 %1, label %BaseCase, label %RecurseCase

BaseCase:
  ret i63 1

RecurseCase:
  %n2 = sub i63 %n, 2
  %n1 = sub i63 %n, 1
  %f2 = call i63(i63) * @fib(i63 %n2)
  %f1 = call i63(i63) * @fib(i63 %n1)
  %result = add i63 %f2, %f1
  ret i63 %result
end

define i63 @"realmain"(i32 %argc, i8 ** %argv)
begin
  icmp eq i32 %argc, 2      ; {i1}:1
  br i1 %1, label %HasArg, label %Continue
HasArg:
  ; %n1 = atoi(argv[1])
  %n1 = add i32 1, 1
  br label %Continue

Continue:
  %n = phi i32 [%n1, %HasArg], [1, %0]
  %N = sext i32 %n to i63
  %F = call i63(i63) *@fib(i63 %N)
  ret i63 %F
end

define i63 @"trampoline"(i63 %n, i63(i63)* %fibfunc)
begin
  %F = call i63(i63) *%fibfunc(i63 %n)
  ret i63 %F
end

define i32 @"main"()
begin
  %Result = call i63 @trampoline(i63 10, i63(i63) *@fib)
  %Result2 = trunc i63 %Result to i32
  ret i32 %Result2
end
