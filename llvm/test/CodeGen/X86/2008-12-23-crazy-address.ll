; RUN: llc < %s -march=x86 -relocation-model=static | grep "lea.*X.*esp" | count 2

@X = external global [0 x i32]

define void @foo() nounwind {
entry:
	%Y = alloca i32
	call void @frob(i32* %Y) nounwind
	%Y3 = bitcast i32* %Y to i8*
	%ctg2 = getelementptr i8, i8* %Y3, i32 ptrtoint ([0 x i32]* @X to i32)
	%0 = ptrtoint i8* %ctg2 to i32
	call void @borf(i32 %0) nounwind
	ret void
}

define void @bar(i32 %i) nounwind {
entry:
	%Y = alloca [10 x i32]
	%0 = getelementptr [10 x i32], [10 x i32]* %Y, i32 0, i32 0
	call void @frob(i32* %0) nounwind
	%1 = getelementptr [0 x i32], [0 x i32]* @X, i32 0, i32 %i
	%2 = getelementptr [10 x i32], [10 x i32]* %Y, i32 0, i32 0
	%3 = ptrtoint i32* %2 to i32
	%4 = bitcast i32* %1 to i8*
	%ctg2 = getelementptr i8, i8* %4, i32 %3
	%5 = ptrtoint i8* %ctg2 to i32
	call void @borf(i32 %5) nounwind
	ret void
}

declare void @frob(i32*)

declare void @borf(i32)
