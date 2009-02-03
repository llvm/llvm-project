; RUN: llvm-as < %s | llc -mtriple=i386-apple-darwin -relocation-model=pic -disable-fp-elim | grep addb | grep ebp

	%struct.rc4_state = type { i32, i32, [256 x i32] }
@.str1 = internal constant [65 x i8] c"m[%d] = 0x%02x, m[%d] = 0x%02x, 0x%02x, k = %d, key[k] = 0x%02x\0A\00"		; <[65 x i8]*> [#uses=1]
@keys = internal constant [7 x [30 x i8]] [ [30 x i8] c"\08\01#Eg\89\AB\CD\EF\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00", [30 x i8] c"\08\01#Eg\89\AB\CD\EF\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00", [30 x i8] c"\08\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00", [30 x i8] c"\04\EF\01#E\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00", [30 x i8] c"\08\01#Eg\89\AB\CD\EF\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00", [30 x i8] c"\04\EF\01#E\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00", [30 x i8] zeroinitializer ]		; <[7 x [30 x i8]]*> [#uses=1]

declare i32 @printf(i8*, ...) nounwind 

define i32 @main(i32 %argc, i8** %argv) nounwind  {
entry:
	br label %bb25

bb25:		; preds = %bb25, %entry
  %foo = phi i1 [ 0, %bb25], [ 1, %entry]
	br i1 %foo, label %bb.i, label %bb25

bb.i:		; preds = %bb.i, %bb25
  %foo2 = phi i1 [ 0, %bb.i], [1, %bb25]
	br i1 %foo2, label %bb21.i, label %bb.i

bb21.i:		; preds = %bb21.i, %bb.i
	%k.0.reg2mem.0.i = phi i32 [ %k.1.i, %bb21.i ], [ 0, %bb.i ]		; <i32> [#uses=2]
	%j.0.reg2mem.0.i = phi i8 [ %tmp35.i, %bb21.i ], [ 0, %bb.i ]		; <i8> [#uses=1]
	%tmp25.i = load i32* null, align 4		; <i32> [#uses=4]
	%tmp2829.i = trunc i32 %tmp25.i to i8		; <i8> [#uses=1]
	%.sum = add i32 %k.0.reg2mem.0.i, 1		; <i32> [#uses=3]
	%tmp33.i = getelementptr [7 x [30 x i8]]* @keys, i32 0, i32 0, i32 %.sum		; <i8*> [#uses=1]
	%tmp34.i = load i8* %tmp33.i, align 1		; <i8> [#uses=1]
	%tmp30.i = add i8 %tmp2829.i, %j.0.reg2mem.0.i		; <i8> [#uses=1]
	%tmp35.i = add i8 %tmp30.i, %tmp34.i		; <i8> [#uses=2]
	%tmp3536.i = zext i8 %tmp35.i to i32		; <i32> [#uses=2]
	%tmp39.i = getelementptr %struct.rc4_state* null, i32 0, i32 2, i32 %tmp3536.i		; <i32*> [#uses=1]
	store i32 %tmp25.i, i32* %tmp39.i, align 4
	%tmp60.i = load i32* null, align 4		; <i32> [#uses=1]
	%tmp65.i = call i32 (i8*, ...)* @printf( i8* getelementptr ([65 x i8]* @.str1, i32 0, i32 0), i32 0, i32 %tmp60.i, i32 %tmp3536.i, i32 %tmp25.i, i32 %tmp25.i, i32 %k.0.reg2mem.0.i, i32 0 ) nounwind 		; <i32> [#uses=0]
	%tmp70.i = icmp slt i32 %.sum, 8		; <i1> [#uses=1]
	%k.1.i = select i1 %tmp70.i, i32 %.sum, i32 0		; <i32> [#uses=1]
	br label %bb21.i
}
