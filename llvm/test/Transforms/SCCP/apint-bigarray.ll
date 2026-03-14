; RUN: opt < %s -passes=sccp -S | not grep %X

@G =  global [1000000 x i10000] zeroinitializer

define internal ptr @test(i10000 %Arg) {
	%X = getelementptr [1000000 x i10000], ptr @G, i32 0, i32 999
        store i10000 %Arg, ptr %X
	ret ptr %X
}

define i10000 @caller()
{
        %Y = call ptr @test(i10000 -1)
        %Z = load i10000, ptr %Y
        ret i10000 %Z 
}

define i10000 @caller2()
{
        %Y = call ptr @test(i10000 1)
        %Z = load i10000, ptr %Y
        ret i10000 %Z 
}
