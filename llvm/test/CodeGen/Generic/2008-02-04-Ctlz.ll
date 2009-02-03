; RUN: llvm-as < %s | llc

@.str = internal constant [14 x i8] c"%lld %d %d %d\00"

define i32 @main(i64 %arg) nounwind  {
entry:
	%tmp37 = tail call i64 @llvm.ctlz.i64( i64 %arg )		; <i64> [#uses=1]
	%tmp47 = tail call i64 @llvm.cttz.i64( i64 %arg )		; <i64> [#uses=1]
	%tmp57 = tail call i64 @llvm.ctpop.i64( i64 %arg )		; <i64> [#uses=1]
	%tmp38 = trunc i64 %tmp37 to i32		; <i32>:0 [#uses=1]
	%tmp48 = trunc i64 %tmp47 to i32		; <i32>:0 [#uses=1]
	%tmp58 = trunc i64 %tmp57 to i32		; <i32>:0 [#uses=1]
	%tmp40 = tail call i32 (i8*, ...)* @printf( i8* noalias  getelementptr ([14 x i8]* @.str, i32 0, i32 0), i64 %arg, i32 %tmp38, i32 %tmp48, i32 %tmp58 ) nounwind 		; <i32> [#uses=0]
	ret i32 0
}

declare i32 @printf(i8* noalias , ...) nounwind 

declare i64 @llvm.ctlz.i64(i64) nounwind readnone 
declare i64 @llvm.cttz.i64(i64) nounwind readnone 
declare i64 @llvm.ctpop.i64(i64) nounwind readnone 
