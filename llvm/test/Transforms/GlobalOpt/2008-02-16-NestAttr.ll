; RUN: llvm-as < %s | opt -globalopt | llvm-dis | grep { nest } | count 1
	%struct.FRAME.nest = type { i32, i32 (i32)* }
	%struct.__builtin_trampoline = type { [10 x i8] }
@.str = internal constant [7 x i8] c"%d %d\0A\00"		; <[7 x i8]*> [#uses=1]

define i32 @process(i32 (i32)* %func) nounwind  {
entry:
	%tmp2 = tail call i32 %func( i32 1 ) nounwind 		; <i32> [#uses=1]
	ret i32 %tmp2
}

define internal fastcc i32 @g.1478(%struct.FRAME.nest* nest  %CHAIN.1, i32 %m) nounwind  {
entry:
	%tmp3 = getelementptr %struct.FRAME.nest* %CHAIN.1, i32 0, i32 0		; <i32*> [#uses=1]
	%tmp4 = load i32* %tmp3, align 4		; <i32> [#uses=1]
	%tmp7 = icmp eq i32 %tmp4, %m		; <i1> [#uses=1]
	%tmp78 = zext i1 %tmp7 to i32		; <i32> [#uses=1]
	ret i32 %tmp78
}

define internal i32 @f.1481(%struct.FRAME.nest* nest  %CHAIN.2, i32 %m) nounwind  {
entry:
	%tmp4 = tail call fastcc i32 @g.1478( %struct.FRAME.nest* nest  %CHAIN.2, i32 %m ) nounwind 		; <i32> [#uses=1]
	%tmp6 = getelementptr %struct.FRAME.nest* %CHAIN.2, i32 0, i32 0		; <i32*> [#uses=1]
	%tmp7 = load i32* %tmp6, align 4		; <i32> [#uses=1]
	%tmp9 = icmp eq i32 %tmp4, %tmp7		; <i1> [#uses=1]
	%tmp910 = zext i1 %tmp9 to i32		; <i32> [#uses=1]
	ret i32 %tmp910
}

define i32 @nest(i32 %n) nounwind  {
entry:
	%TRAMP.316 = alloca [10 x i8]		; <[10 x i8]*> [#uses=1]
	%FRAME.0 = alloca %struct.FRAME.nest		; <%struct.FRAME.nest*> [#uses=3]
	%TRAMP.316.sub = getelementptr [10 x i8]* %TRAMP.316, i32 0, i32 0		; <i8*> [#uses=1]
	%tmp3 = getelementptr %struct.FRAME.nest* %FRAME.0, i32 0, i32 0		; <i32*> [#uses=1]
	store i32 %n, i32* %tmp3, align 8
	%FRAME.06 = bitcast %struct.FRAME.nest* %FRAME.0 to i8*		; <i8*> [#uses=1]
	%tramp = call i8* @llvm.init.trampoline( i8* %TRAMP.316.sub, i8* bitcast (i32 (%struct.FRAME.nest*, i32)* @f.1481 to i8*), i8* %FRAME.06 )		; <i8*> [#uses=1]
	%tmp7 = getelementptr %struct.FRAME.nest* %FRAME.0, i32 0, i32 1		; <i32 (i32)**> [#uses=1]
	%tmp89 = bitcast i8* %tramp to i32 (i32)*		; <i32 (i32)*> [#uses=2]
	store i32 (i32)* %tmp89, i32 (i32)** %tmp7, align 4
	%tmp13 = call i32 @process( i32 (i32)* %tmp89 ) nounwind 		; <i32> [#uses=1]
	ret i32 %tmp13
}

declare i8* @llvm.init.trampoline(i8*, i8*, i8*) nounwind 

define i32 @main() nounwind  {
entry:
	%tmp = tail call i32 @nest( i32 2 ) nounwind 		; <i32> [#uses=1]
	%tmp1 = tail call i32 @nest( i32 1 ) nounwind 		; <i32> [#uses=1]
	%tmp3 = tail call i32 (i8*, ...)* @printf( i8* noalias  getelementptr ([7 x i8]* @.str, i32 0, i32 0), i32 %tmp1, i32 %tmp ) nounwind 		; <i32> [#uses=0]
	ret i32 undef
}

declare i32 @printf(i8*, ...) nounwind 
