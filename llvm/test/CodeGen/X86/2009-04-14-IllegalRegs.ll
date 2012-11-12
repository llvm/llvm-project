; RUN: llc < %s -mtriple=i386-apple-darwin -O0 -regalloc=fast | not grep sil
; rdar://6787136

	%struct.X = type { i8, [32 x i8] }
@llvm.used = appending global [1 x i8*] [i8* bitcast (i32 ()* @z to i8*)], section "llvm.metadata"		; <[1 x i8*]*> [#uses=0]

define i32 @z() nounwind ssp {
entry:
	%retval = alloca i32		; <i32*> [#uses=2]
	%xxx = alloca %struct.X		; <%struct.X*> [#uses=6]
	%0 = alloca i32		; <i32*> [#uses=2]
	%"alloca point" = bitcast i32 0 to i32		; <i32> [#uses=0]
	%1 = getelementptr %struct.X* %xxx, i32 0, i32 1		; <[32 x i8]*> [#uses=1]
	%2 = getelementptr [32 x i8]* %1, i32 0, i32 31		; <i8*> [#uses=1]
	store i8 48, i8* %2, align 1
	%3 = getelementptr %struct.X* %xxx, i32 0, i32 1		; <[32 x i8]*> [#uses=1]
	%4 = getelementptr [32 x i8]* %3, i32 0, i32 31		; <i8*> [#uses=1]
	%5 = load i8* %4, align 1		; <i8> [#uses=1]
	%6 = getelementptr %struct.X* %xxx, i32 0, i32 1		; <[32 x i8]*> [#uses=1]
	%7 = getelementptr [32 x i8]* %6, i32 0, i32 0		; <i8*> [#uses=1]
	store i8 %5, i8* %7, align 1
	%8 = getelementptr %struct.X* %xxx, i32 0, i32 0		; <i8*> [#uses=1]
	store i8 15, i8* %8, align 1
	%9 = call i32 (...)* bitcast (i32 (%struct.X*, %struct.X*)* @f to i32 (...)*)(%struct.X* byval align 4 %xxx, %struct.X* byval align 4 %xxx) nounwind		; <i32> [#uses=1]
	store i32 %9, i32* %0, align 4
	%10 = load i32* %0, align 4		; <i32> [#uses=1]
	store i32 %10, i32* %retval, align 4
	br label %return

return:		; preds = %entry
	%retval1 = load i32* %retval		; <i32> [#uses=1]
	ret i32 %retval1
}

declare i32 @f(%struct.X* byval align 4, %struct.X* byval align 4) nounwind ssp
