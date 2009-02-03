; RUN: llvm-as %s -o - | llvm-dis > %t1.ll
; RUN: llvm-as %t1.ll -o - | llvm-dis > %t2.ll
; RUN: diff %t1.ll %t2.ll

%FunTy = type i31(i31)


define void @"invoke"(%FunTy *%x)
begin
	%foo = call %FunTy* %x(i31 123)
	ret void
end

define i31 @"main"(i31 %argc, i8 **%argv, i8 **%envp)
begin
        %retval = call i31 (i31) *@test(i31 %argc)
        %two    = add i31 %retval, %retval
	%retval2 = call i31 @test(i31 %argc)

	%two2 = add i31 %two, %retval2
	call void @invoke (%FunTy* @test)
        ret i31 %two2
end

define i31 @"test"(i31 %i0)
begin
    ret i31 %i0
end
