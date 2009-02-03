; RUN: llvm-as < %s | opt -basicaa -gvn | llvm-dis | grep TestConst | count 2
; RUN: llvm-as < %s | opt -basicaa -gvn | llvm-dis | grep TestPure  | count 3
; RUN: llvm-as < %s | opt -basicaa -gvn | llvm-dis | grep TestNone  | count 4
@g = global i32 0		; <i32*> [#uses=1]

define i32 @test() {
entry:
	%tmp0 = call i32 @TestConst( i32 5 ) readnone 		; <i32> [#uses=1]
	%tmp1 = call i32 @TestPure( i32 6 ) readonly 		; <i32> [#uses=1]
	%tmp2 = call i32 @TestNone( i32 7 )		; <i32> [#uses=1]
	store i32 1, i32* @g
	%tmp3 = call i32 @TestConst( i32 5 ) readnone 		; <i32> [#uses=1]
	%tmp4 = call i32 @TestConst( i32 5 ) readnone 		; <i32> [#uses=1]
	%tmp5 = call i32 @TestPure( i32 6 ) readonly 		; <i32> [#uses=1]
	%tmp6 = call i32 @TestPure( i32 6 ) readonly 		; <i32> [#uses=1]
	%tmp7 = call i32 @TestNone( i32 7 )		; <i32> [#uses=1]
	%tmp8 = call i32 @TestNone( i32 7 )		; <i32> [#uses=1]
	%sum0 = add i32 %tmp0, %tmp1		; <i32> [#uses=1]
	%sum1 = add i32 %sum0, %tmp2		; <i32> [#uses=1]
	%sum2 = add i32 %sum1, %tmp3		; <i32> [#uses=1]
	%sum3 = add i32 %sum2, %tmp4		; <i32> [#uses=1]
	%sum4 = add i32 %sum3, %tmp5		; <i32> [#uses=1]
	%sum5 = add i32 %sum4, %tmp6		; <i32> [#uses=1]
	%sum6 = add i32 %sum5, %tmp7		; <i32> [#uses=1]
	%sum7 = add i32 %sum6, %tmp8		; <i32> [#uses=1]
	ret i32 %sum7
}

declare i32 @TestConst(i32) readnone

declare i32 @TestPure(i32) readonly

declare i32 @TestNone(i32)
