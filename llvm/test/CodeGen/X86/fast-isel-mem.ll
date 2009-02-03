; RUN: llvm-as < %s | llc -fast-isel -mtriple=i386-apple-darwin | \
; RUN:   grep lazy_ptr, | count 2
; RUN: llvm-as < %s | llc -fast-isel -march=x86 -relocation-model=static | \
; RUN:   grep lea

@src = external global i32

define i32 @loadgv() nounwind {
entry:
	%0 = load i32* @src, align 4
	%1 = load i32* @src, align 4
        %2 = add i32 %0, %1
        store i32 %2, i32* @src
	ret i32 %2
}

%stuff = type { i32 (...)** }
@LotsStuff = external constant [4 x i32 (...)*]

define void @t(%stuff* %this) nounwind {
entry:
	store i32 (...)** getelementptr ([4 x i32 (...)*]* @LotsStuff, i32 0, i32 2), i32 (...)*** null, align 4
	ret void
}
