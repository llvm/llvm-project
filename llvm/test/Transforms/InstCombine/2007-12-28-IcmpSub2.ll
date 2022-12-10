; RUN: opt < %s -passes=mem2reg,instcombine -S | grep "ret i32 1" | count 8

define i32 @test1() {
entry:
	%z = alloca i32
	store i32 0, ptr %z
	%tmp = load i32, ptr %z
	%sub = sub i32 %tmp, 1
	%cmp = icmp ule i32 %sub, 0
	%retval = select i1 %cmp, i32 0, i32 1
	ret i32 %retval
}

define i32 @test2() {
entry:
	%z = alloca i32
	store i32 0, ptr %z
	%tmp = load i32, ptr %z
	%sub = sub i32 %tmp, 1
	%cmp = icmp ugt i32 %sub, 0
	%retval = select i1 %cmp, i32 1, i32 0
	ret i32 %retval
}

define i32 @test3() {
entry:
	%z = alloca i32
	store i32 0, ptr %z
	%tmp = load i32, ptr %z
	%sub = sub i32 %tmp, 1
	%cmp = icmp slt i32 %sub, 0
	%retval = select i1 %cmp, i32 1, i32 0
	ret i32 %retval
}

define i32 @test4() {
entry:
	%z = alloca i32
	store i32 0, ptr %z
	%tmp = load i32, ptr %z
	%sub = sub i32 %tmp, 1
	%cmp = icmp sle i32 %sub, 0
	%retval = select i1 %cmp, i32 1, i32 0
	ret i32 %retval
}

define i32 @test5() {
entry:
	%z = alloca i32
	store i32 0, ptr %z
	%tmp = load i32, ptr %z
	%sub = sub i32 %tmp, 1
	%cmp = icmp sge i32 %sub, 0
	%retval = select i1 %cmp, i32 0, i32 1
	ret i32 %retval
}

define i32 @test6() {
entry:
	%z = alloca i32
	store i32 0, ptr %z
	%tmp = load i32, ptr %z
	%sub = sub i32 %tmp, 1
	%cmp = icmp sgt i32 %sub, 0
	%retval = select i1 %cmp, i32 0, i32 1
	ret i32 %retval
}

define i32 @test7() {
entry:
	%z = alloca i32
	store i32 0, ptr %z
	%tmp = load i32, ptr %z
	%sub = sub i32 %tmp, 1
	%cmp = icmp eq i32 %sub, 0
	%retval = select i1 %cmp, i32 0, i32 1
	ret i32 %retval
}

define i32 @test8() {
entry:
	%z = alloca i32
	store i32 0, ptr %z
	%tmp = load i32, ptr %z
	%sub = sub i32 %tmp, 1
	%cmp = icmp ne i32 %sub, 0
	%retval = select i1 %cmp, i32 1, i32 0
	ret i32 %retval
}
