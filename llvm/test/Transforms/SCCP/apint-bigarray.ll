; RUN: opt < %s -sccp -S | not grep %X

@G =  global [1000000 x i10000] zeroinitializer

define internal i10000* @test(i10000 %Arg) {
	%X = getelementptr [1000000 x i10000], [1000000 x i10000]* @G, i32 0, i32 999
        store i10000 %Arg, i10000* %X
	ret i10000* %X
}

define i10000 @caller()
{
        %Y = call i10000* @test(i10000 -1)
        %Z = load i10000, i10000* %Y
        ret i10000 %Z 
}

define i10000 @caller2()
{
        %Y = call i10000* @test(i10000 1)
        %Z = load i10000, i10000* %Y
        ret i10000 %Z 
}
