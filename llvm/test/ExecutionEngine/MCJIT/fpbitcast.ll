; RUN: %lli -jit-kind=mcjit -force-interpreter=true %s | FileCheck %s
; CHECK: 40091eb8

define i32 @test(double %x) {
entry:
	%x46.i = bitcast double %x to i64	
	%tmp343.i = lshr i64 %x46.i, 32	
	%tmp344.i = trunc i64 %tmp343.i to i32
        ret i32 %tmp344.i
}

define i32 @main()
{
       %res = call i32 @test(double 3.14)
       call i32 (ptr,...) @printf(ptr @format, i32 %res)
       ret i32 0
}

declare i32 @printf(ptr, ...)
@format = internal constant [4 x i8] c"%x\0A\00"
