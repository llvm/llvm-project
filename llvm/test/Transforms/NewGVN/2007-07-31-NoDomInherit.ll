; XFAIL: *
; RUN: opt < %s -basic-aa -newgvn -S | FileCheck %s

	%struct.anon = type { i32 (i32, i32, i32)*, i32, i32, [3 x i32], i8*, i8*, i8* }
@debug = external constant i32		; <i32*> [#uses=0]
@counters = external constant i32		; <i32*> [#uses=1]
@trialx = external global [17 x i32]		; <[17 x i32]*> [#uses=1]
@dummy1 = external global [7 x i32]		; <[7 x i32]*> [#uses=0]
@dummy2 = external global [4 x i32]		; <[4 x i32]*> [#uses=0]
@unacceptable = external global i32		; <i32*> [#uses=0]
@isa = external global [13 x %struct.anon]		; <[13 x %struct.anon]*> [#uses=3]
@.str = external constant [4 x i8]		; <[4 x i8]*> [#uses=0]
@.str1 = external constant [3 x i8]		; <[3 x i8]*> [#uses=0]
@.str2 = external constant [1 x i8]		; <[1 x i8]*> [#uses=0]
@.str3 = external constant [4 x i8]		; <[4 x i8]*> [#uses=0]
@.str4 = external constant [3 x i8]		; <[3 x i8]*> [#uses=0]
@.str5 = external constant [4 x i8]		; <[4 x i8]*> [#uses=0]
@.str6 = external constant [2 x i8]		; <[2 x i8]*> [#uses=0]
@.str7 = external constant [4 x i8]		; <[4 x i8]*> [#uses=0]
@.str8 = external constant [4 x i8]		; <[4 x i8]*> [#uses=0]
@.str9 = external constant [4 x i8]		; <[4 x i8]*> [#uses=0]
@.str10 = external constant [4 x i8]		; <[4 x i8]*> [#uses=0]
@.str11 = external constant [2 x i8]		; <[2 x i8]*> [#uses=0]
@.str12 = external constant [4 x i8]		; <[4 x i8]*> [#uses=0]
@.str13 = external constant [2 x i8]		; <[2 x i8]*> [#uses=0]
@.str14 = external constant [5 x i8]		; <[5 x i8]*> [#uses=0]
@.str15 = external constant [5 x i8]		; <[5 x i8]*> [#uses=0]
@.str16 = external constant [4 x i8]		; <[4 x i8]*> [#uses=0]
@.str17 = external constant [4 x i8]		; <[4 x i8]*> [#uses=0]
@.str18 = external constant [3 x i8]		; <[3 x i8]*> [#uses=0]
@.str19 = external constant [4 x i8]		; <[4 x i8]*> [#uses=0]
@.str20 = external constant [4 x i8]		; <[4 x i8]*> [#uses=0]
@.str21 = external constant [4 x i8]		; <[4 x i8]*> [#uses=0]
@.str22 = external constant [4 x i8]		; <[4 x i8]*> [#uses=0]
@.str23 = external constant [5 x i8]		; <[5 x i8]*> [#uses=0]
@.str24 = external constant [4 x i8]		; <[4 x i8]*> [#uses=0]
@.str25 = external constant [6 x i8]		; <[6 x i8]*> [#uses=0]
@.str26 = external constant [5 x i8]		; <[5 x i8]*> [#uses=0]
@.str27 = external constant [6 x i8]		; <[6 x i8]*> [#uses=0]
@r = external global [17 x i32]		; <[17 x i32]*> [#uses=0]
@.str28 = external constant [3 x i8]		; <[3 x i8]*> [#uses=0]
@.str29 = external constant [5 x i8]		; <[5 x i8]*> [#uses=0]
@pgm = external global [5 x { i32, [3 x i32] }]		; <[5 x { i32, [3 x i32] }]*> [#uses=4]
@.str30 = external constant [3 x i8]		; <[3 x i8]*> [#uses=0]
@.str31 = external constant [13 x i8]		; <[13 x i8]*> [#uses=0]
@.str32 = external constant [3 x i8]		; <[3 x i8]*> [#uses=0]
@.str33 = external constant [4 x i8]		; <[4 x i8]*> [#uses=0]
@.str34 = external constant [20 x i8]		; <[20 x i8]*> [#uses=0]
@numi = external global i32		; <i32*> [#uses=7]
@.str35 = external constant [10 x i8]		; <[10 x i8]*> [#uses=0]
@counter = external global [5 x i32]		; <[5 x i32]*> [#uses=2]
@itrialx.2510 = external global i32		; <i32*> [#uses=0]
@.str36 = external constant [43 x i8]		; <[43 x i8]*> [#uses=0]
@.str37 = external constant [42 x i8]		; <[42 x i8]*> [#uses=0]
@corr_result = external global i32		; <i32*> [#uses=0]
@.str38 = external constant [3 x i8]		; <[3 x i8]*> [#uses=0]
@.str39 = external constant [5 x i8]		; <[5 x i8]*> [#uses=0]
@.str40 = external constant [47 x i8]		; <[47 x i8]*> [#uses=0]
@correct_result = external global [17 x i32]		; <[17 x i32]*> [#uses=1]
@.str41 = external constant [46 x i8]		; <[46 x i8]*> [#uses=0]
@.str42 = external constant [32 x i8]		; <[32 x i8]*> [#uses=0]
@.str43 = external constant [44 x i8]		; <[44 x i8]*> [#uses=1]
@.str44 = external constant [21 x i8]		; <[21 x i8]*> [#uses=1]
@.str45 = external constant [12 x i8]		; <[12 x i8]*> [#uses=1]
@.str46 = external constant [5 x i8]		; <[5 x i8]*> [#uses=1]
@.str47 = external constant [12 x i8]		; <[12 x i8]*> [#uses=1]

declare i32 @neg(i32, i32, i32)

declare i32 @Not(i32, i32, i32)

declare i32 @pop(i32, i32, i32)

declare i32 @nlz(i32, i32, i32)

declare i32 @rev(i32, i32, i32)

declare i32 @add(i32, i32, i32)

declare i32 @sub(i32, i32, i32)

declare i32 @mul(i32, i32, i32)

declare i32 @divide(i32, i32, i32)

declare i32 @divu(i32, i32, i32)

declare i32 @And(i32, i32, i32)

declare i32 @Or(i32, i32, i32)

declare i32 @Xor(i32, i32, i32)

declare i32 @rotl(i32, i32, i32)

declare i32 @shl(i32, i32, i32)

declare i32 @shr(i32, i32, i32)

declare i32 @shrs(i32, i32, i32)

declare i32 @cmpeq(i32, i32, i32)

declare i32 @cmplt(i32, i32, i32)

declare i32 @cmpltu(i32, i32, i32)

declare i32 @seleq(i32, i32, i32)

declare i32 @sellt(i32, i32, i32)

declare i32 @selle(i32, i32, i32)

declare void @print_expr(i32)

declare i32 @printf(i8*, ...)

declare i32 @putchar(i32)

declare void @print_pgm()

declare void @simulate_one_instruction(i32)

declare i32 @check(i32)

declare i32 @puts(i8*)

declare void @fix_operands(i32)

declare void @abort()

declare i32 @increment()

declare i32 @search()

define i32 @main(i32 %argc, i8** %argv) {
entry:
	%argc_addr = alloca i32		; <i32*> [#uses=1]
	%argv_addr = alloca i8**		; <i8***> [#uses=1]
	%retval = alloca i32, align 4		; <i32*> [#uses=2]
	%tmp = alloca i32, align 4		; <i32*> [#uses=2]
	%i = alloca i32, align 4		; <i32*> [#uses=21]
	%num_sol = alloca i32, align 4		; <i32*> [#uses=4]
	%total = alloca i32, align 4		; <i32*> [#uses=4]
	%"alloca point" = bitcast i32 0 to i32		; <i32> [#uses=0]
	store i32 %argc, i32* %argc_addr
	store i8** %argv, i8*** %argv_addr
	store i32 0, i32* %num_sol
	store i32 1, i32* @numi
	br label %bb91

bb:		; preds = %cond_next97
	%tmp1 = load i32, i32* @numi		; <i32> [#uses=1]
	%tmp2 = getelementptr [44 x i8], [44 x i8]* @.str43, i32 0, i32 0		; <i8*> [#uses=1]
	%tmp3 = call i32 (i8*, ...) @printf( i8* %tmp2, i32 %tmp1 )		; <i32> [#uses=0]
	store i32 0, i32* %i
	br label %bb13

bb4:		; preds = %bb13
	%tmp5 = load i32, i32* %i		; <i32> [#uses=1]
	%tmp6 = load i32, i32* %i		; <i32> [#uses=1]
	%tmp7 = getelementptr [17 x i32], [17 x i32]* @trialx, i32 0, i32 %tmp6		; <i32*> [#uses=1]
	%tmp8 = load i32, i32* %tmp7		; <i32> [#uses=1]
	%tmp9 = call i32 @userfun( i32 %tmp8 )		; <i32> [#uses=1]
	%tmp10 = getelementptr [17 x i32], [17 x i32]* @correct_result, i32 0, i32 %tmp5		; <i32*> [#uses=1]
	store i32 %tmp9, i32* %tmp10
	%tmp11 = load i32, i32* %i		; <i32> [#uses=1]
	%tmp12 = add i32 %tmp11, 1		; <i32> [#uses=1]
	store i32 %tmp12, i32* %i
	br label %bb13

bb13:		; preds = %bb4, %bb
	%tmp14 = load i32, i32* %i		; <i32> [#uses=1]
	%tmp15 = icmp sle i32 %tmp14, 16		; <i1> [#uses=1]
	%tmp1516 = zext i1 %tmp15 to i32		; <i32> [#uses=1]
	%toBool = icmp ne i32 %tmp1516, 0		; <i1> [#uses=1]
	br i1 %toBool, label %bb4, label %bb17

bb17:		; preds = %bb13
	store i32 0, i32* %i
	br label %bb49

bb18:		; preds = %bb49
	%tmp19 = load i32, i32* %i		; <i32> [#uses=1]
	%tmp20 = getelementptr [5 x { i32, [3 x i32] }], [5 x { i32, [3 x i32] }]* @pgm, i32 0, i32 %tmp19		; <{ i32, [3 x i32] }*> [#uses=1]
	%tmp21 = getelementptr { i32, [3 x i32] }, { i32, [3 x i32] }* %tmp20, i32 0, i32 0		; <i32*> [#uses=1]
	store i32 0, i32* %tmp21
	%tmp22 = load i32, i32* %i		; <i32> [#uses=1]
	%tmp23 = getelementptr [13 x %struct.anon], [13 x %struct.anon]* @isa, i32 0, i32 0		; <%struct.anon*> [#uses=1]
	%tmp24 = getelementptr %struct.anon, %struct.anon* %tmp23, i32 0, i32 3		; <[3 x i32]*> [#uses=1]
	%tmp25 = getelementptr [3 x i32], [3 x i32]* %tmp24, i32 0, i32 0		; <i32*> [#uses=1]
	%tmp26 = load i32, i32* %tmp25		; <i32> [#uses=1]
	%tmp27 = getelementptr [5 x { i32, [3 x i32] }], [5 x { i32, [3 x i32] }]* @pgm, i32 0, i32 %tmp22		; <{ i32, [3 x i32] }*> [#uses=1]
	%tmp28 = getelementptr { i32, [3 x i32] }, { i32, [3 x i32] }* %tmp27, i32 0, i32 1		; <[3 x i32]*> [#uses=1]
	%tmp29 = getelementptr [3 x i32], [3 x i32]* %tmp28, i32 0, i32 0		; <i32*> [#uses=1]
	store i32 %tmp26, i32* %tmp29
	%tmp30 = load i32, i32* %i		; <i32> [#uses=1]
	%tmp31 = getelementptr [13 x %struct.anon], [13 x %struct.anon]* @isa, i32 0, i32 0		; <%struct.anon*> [#uses=1]
	%tmp32 = getelementptr %struct.anon, %struct.anon* %tmp31, i32 0, i32 3		; <[3 x i32]*> [#uses=1]
	%tmp33 = getelementptr [3 x i32], [3 x i32]* %tmp32, i32 0, i32 1		; <i32*> [#uses=1]
	%tmp34 = load i32, i32* %tmp33		; <i32> [#uses=1]
	%tmp35 = getelementptr [5 x { i32, [3 x i32] }], [5 x { i32, [3 x i32] }]* @pgm, i32 0, i32 %tmp30		; <{ i32, [3 x i32] }*> [#uses=1]
	%tmp36 = getelementptr { i32, [3 x i32] }, { i32, [3 x i32] }* %tmp35, i32 0, i32 1		; <[3 x i32]*> [#uses=1]
	%tmp37 = getelementptr [3 x i32], [3 x i32]* %tmp36, i32 0, i32 1		; <i32*> [#uses=1]
	store i32 %tmp34, i32* %tmp37
	%tmp38 = load i32, i32* %i		; <i32> [#uses=1]
	%tmp39 = getelementptr [13 x %struct.anon], [13 x %struct.anon]* @isa, i32 0, i32 0		; <%struct.anon*> [#uses=1]
	%tmp40 = getelementptr %struct.anon, %struct.anon* %tmp39, i32 0, i32 3		; <[3 x i32]*> [#uses=1]
	%tmp41 = getelementptr [3 x i32], [3 x i32]* %tmp40, i32 0, i32 2		; <i32*> [#uses=1]
	%tmp42 = load i32, i32* %tmp41		; <i32> [#uses=1]
	%tmp43 = getelementptr [5 x { i32, [3 x i32] }], [5 x { i32, [3 x i32] }]* @pgm, i32 0, i32 %tmp38		; <{ i32, [3 x i32] }*> [#uses=1]
	%tmp44 = getelementptr { i32, [3 x i32] }, { i32, [3 x i32] }* %tmp43, i32 0, i32 1		; <[3 x i32]*> [#uses=1]
	%tmp45 = getelementptr [3 x i32], [3 x i32]* %tmp44, i32 0, i32 2		; <i32*> [#uses=1]
	store i32 %tmp42, i32* %tmp45
	%tmp46 = load i32, i32* %i		; <i32> [#uses=1]
	call void @fix_operands( i32 %tmp46 )
	%tmp47 = load i32, i32* %i		; <i32> [#uses=1]
; CHECK: %tmp47 = phi i32 [ %tmp48, %bb18 ], [ 0, %bb17 ]
	%tmp48 = add i32 %tmp47, 1		; <i32> [#uses=1]
	store i32 %tmp48, i32* %i
	br label %bb49

bb49:		; preds = %bb18, %bb17
	%tmp50 = load i32, i32* @numi		; <i32> [#uses=1]
	%tmp51 = load i32, i32* %i		; <i32> [#uses=1]
	%tmp52 = icmp slt i32 %tmp51, %tmp50		; <i1> [#uses=1]
	%tmp5253 = zext i1 %tmp52 to i32		; <i32> [#uses=1]
	%toBool54 = icmp ne i32 %tmp5253, 0		; <i1> [#uses=1]
	br i1 %toBool54, label %bb18, label %bb55

bb55:		; preds = %bb49
	%tmp56 = call i32 @search( )		; <i32> [#uses=1]
	store i32 %tmp56, i32* %num_sol
	%tmp57 = getelementptr [21 x i8], [21 x i8]* @.str44, i32 0, i32 0		; <i8*> [#uses=1]
	%tmp58 = load i32, i32* %num_sol		; <i32> [#uses=1]
	%tmp59 = call i32 (i8*, ...) @printf( i8* %tmp57, i32 %tmp58 )		; <i32> [#uses=0]
	%tmp60 = load i32, i32* @counters		; <i32> [#uses=1]
	%tmp61 = icmp ne i32 %tmp60, 0		; <i1> [#uses=1]
	%tmp6162 = zext i1 %tmp61 to i32		; <i32> [#uses=1]
	%toBool63 = icmp ne i32 %tmp6162, 0		; <i1> [#uses=1]
	br i1 %toBool63, label %cond_true, label %cond_next

cond_true:		; preds = %bb55
	store i32 0, i32* %total
	%tmp64 = getelementptr [12 x i8], [12 x i8]* @.str45, i32 0, i32 0		; <i8*> [#uses=1]
	%tmp65 = call i32 (i8*, ...) @printf( i8* %tmp64 )		; <i32> [#uses=0]
	store i32 0, i32* %i
	br label %bb79

bb66:		; preds = %bb79
	%tmp67 = load i32, i32* %i		; <i32> [#uses=1]
	%tmp68 = getelementptr [5 x i32], [5 x i32]* @counter, i32 0, i32 %tmp67		; <i32*> [#uses=1]
	%tmp69 = load i32, i32* %tmp68		; <i32> [#uses=1]
	%tmp70 = getelementptr [5 x i8], [5 x i8]* @.str46, i32 0, i32 0		; <i8*> [#uses=1]
	%tmp71 = call i32 (i8*, ...) @printf( i8* %tmp70, i32 %tmp69 )		; <i32> [#uses=0]
	%tmp72 = load i32, i32* %i		; <i32> [#uses=1]
	%tmp73 = getelementptr [5 x i32], [5 x i32]* @counter, i32 0, i32 %tmp72		; <i32*> [#uses=1]
	%tmp74 = load i32, i32* %tmp73		; <i32> [#uses=1]
	%tmp75 = load i32, i32* %total		; <i32> [#uses=1]
	%tmp76 = add i32 %tmp74, %tmp75		; <i32> [#uses=1]
	store i32 %tmp76, i32* %total
	%tmp77 = load i32, i32* %i		; <i32> [#uses=1]
	%tmp78 = add i32 %tmp77, 1		; <i32> [#uses=1]
	store i32 %tmp78, i32* %i
	br label %bb79

bb79:		; preds = %bb66, %cond_true
	%tmp80 = load i32, i32* @numi		; <i32> [#uses=1]
	%tmp81 = load i32, i32* %i		; <i32> [#uses=1]
	%tmp82 = icmp slt i32 %tmp81, %tmp80		; <i1> [#uses=1]
	%tmp8283 = zext i1 %tmp82 to i32		; <i32> [#uses=1]
	%toBool84 = icmp ne i32 %tmp8283, 0		; <i1> [#uses=1]
	br i1 %toBool84, label %bb66, label %bb85

bb85:		; preds = %bb79
	%tmp86 = getelementptr [12 x i8], [12 x i8]* @.str47, i32 0, i32 0		; <i8*> [#uses=1]
	%tmp87 = load i32, i32* %total		; <i32> [#uses=1]
	%tmp88 = call i32 (i8*, ...) @printf( i8* %tmp86, i32 %tmp87 )		; <i32> [#uses=0]
	br label %cond_next

cond_next:		; preds = %bb85, %bb55
	%tmp89 = load i32, i32* @numi		; <i32> [#uses=1]
	%tmp90 = add i32 %tmp89, 1		; <i32> [#uses=1]
	store i32 %tmp90, i32* @numi
	br label %bb91

bb91:		; preds = %cond_next, %entry
	%tmp92 = load i32, i32* @numi		; <i32> [#uses=1]
	%tmp93 = icmp sgt i32 %tmp92, 5		; <i1> [#uses=1]
	%tmp9394 = zext i1 %tmp93 to i32		; <i32> [#uses=1]
	%toBool95 = icmp ne i32 %tmp9394, 0		; <i1> [#uses=1]
	br i1 %toBool95, label %cond_true96, label %cond_next97

cond_true96:		; preds = %bb91
	br label %bb102

cond_next97:		; preds = %bb91
	%tmp98 = load i32, i32* %num_sol		; <i32> [#uses=1]
	%tmp99 = icmp eq i32 %tmp98, 0		; <i1> [#uses=1]
	%tmp99100 = zext i1 %tmp99 to i32		; <i32> [#uses=1]
	%toBool101 = icmp ne i32 %tmp99100, 0		; <i1> [#uses=1]
	br i1 %toBool101, label %bb, label %bb102

bb102:		; preds = %cond_next97, %cond_true96
	store i32 0, i32* %tmp
	%tmp103 = load i32, i32* %tmp		; <i32> [#uses=1]
	store i32 %tmp103, i32* %retval
	br label %return

return:		; preds = %bb102
	%retval104 = load i32, i32* %retval		; <i32> [#uses=1]
	ret i32 %retval104
}

declare i32 @userfun(i32)
