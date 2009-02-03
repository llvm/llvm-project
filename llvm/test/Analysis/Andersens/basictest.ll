; RUN: llvm-as < %s | opt -anders-aa -aa-eval 2>/dev/null

define void @test1() {
	%X = malloc i32*
	%Y = malloc i32
	%Z = ptrtoint i32* %Y to i32
	%W = inttoptr i32 %Z to i32*
	store i32* %W, i32** %X
	ret void
}

define void @test2(i32* %P) {
	%X = malloc i32*
	%Y = malloc i32
	store i32* %P, i32** %X
	ret void
}

define internal i32 *@test3(i32* %P) {
	ret i32* %P
}

define void @test4() {
	%X = malloc i32
	%Y = call i32* @test3(i32* %X)
	%ZZ = getelementptr i32* null, i32 17
	ret void
}
