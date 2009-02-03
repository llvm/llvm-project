; RUN: llvm-as < %s | llvm-dis > /dev/null
; RUN: llvm-as < %s | llc -march=cpp -cppgen=program -o -

@X = global i32 4, align 16		; <i32*> [#uses=0]

define i32* @test1012() align 32 {
	%X = alloca i32, align 4		; <i32*> [#uses=1]
	%Y = alloca i32, i32 42, align 16		; <i32*> [#uses=0]
	%Z = alloca i32		; <i32*> [#uses=0]
	ret i32* %X
}

define i32* @test1013() {
	%X = malloc i32, align 4		; <i32*> [#uses=1]
	%Y = malloc i32, i32 42, align 16		; <i32*> [#uses=0]
	%Z = malloc i32		; <i32*> [#uses=0]
	ret i32* %X
}

define void @void(i32, i32) {
	add i32 0, 0		; <i32>:3 [#uses=2]
	sub i32 0, 4		; <i32>:4 [#uses=2]
	br label %5

; <label>:5		; preds = %5, %2
	add i32 %0, %1		; <i32>:6 [#uses=2]
	sub i32 %6, %4		; <i32>:7 [#uses=1]
	icmp sle i32 %7, %3		; <i1>:8 [#uses=1]
	br i1 %8, label %9, label %5

; <label>:9		; preds = %5
	add i32 %0, %1		; <i32>:10 [#uses=0]
	sub i32 %6, %4		; <i32>:11 [#uses=1]
	icmp sle i32 %11, %3		; <i1>:12 [#uses=0]
	ret void
}

define i32 @zarro() {
Startup:
	ret i32 0
}

define fastcc void @foo() {
	ret void
}

define coldcc void @bar() {
	call fastcc void @foo( )
	ret void
}

define void @structret({ i8 }* sret  %P) {
	call void @structret( { i8 }* %P sret  )
	ret void
}

define void @foo4() {
	ret void
}

define coldcc void @bar2() {
	call fastcc void @foo( )
	ret void
}

define cc42 void @bar3() {
	invoke fastcc void @foo( )
			to label %Ok unwind label %U

Ok:		; preds = %0
	ret void

U:		; preds = %0
	unwind
}

define void @bar4() {
	call cc42 void @bar( )
	invoke cc42 void @bar3( )
			to label %Ok unwind label %U

Ok:		; preds = %0
	ret void

U:		; preds = %0
	unwind
}
; ModuleID = 'calltest.ll'
	%FunTy = type i32 (i32)

define i32 @test1000(i32 %i0) {
	ret i32 %i0
}

define void @invoke(%FunTy* %x) {
	%foo = call i32 %x( i32 123 )		; <i32> [#uses=0]
	%foo2 = tail call i32 %x( i32 123 )		; <i32> [#uses=0]
	ret void
}

define i32 @main(i32 %argc) {
	%retval = call i32 @test1000( i32 %argc )		; <i32> [#uses=2]
	%two = add i32 %retval, %retval		; <i32> [#uses=1]
	%retval2 = invoke i32 @test1000( i32 %argc )
			to label %Next unwind label %Error		; <i32> [#uses=1]

Next:		; preds = %0
	%two2 = add i32 %two, %retval2		; <i32> [#uses=1]
	call void @invoke( %FunTy* @test1000 )
	ret i32 %two2

Error:		; preds = %0
	ret i32 -1
}
; ModuleID = 'casttest.ll'

define i16 @FunFunc(i64 %x, i8 %z) {
bb0:
	%cast110 = sext i8 %z to i16		; <i16> [#uses=1]
	%cast10 = trunc i64 %x to i16		; <i16> [#uses=1]
	%reg109 = add i16 %cast110, %cast10		; <i16> [#uses=1]
	ret i16 %reg109
}
; ModuleID = 'cfgstructures.ll'

define void @irreducible(i1 %cond) {
	br i1 %cond, label %X, label %Y

X:		; preds = %Y, %0
	br label %Y

Y:		; preds = %X, %0
	br label %X
}

define void @sharedheader(i1 %cond) {
	br label %A

A:		; preds = %Y, %X, %0
	br i1 %cond, label %X, label %Y

X:		; preds = %A
	br label %A

Y:		; preds = %A
	br label %A
}

define void @nested(i1 %cond1, i1 %cond2, i1 %cond3) {
	br label %Loop1

Loop1:		; preds = %L2Exit, %0
	br label %Loop2

Loop2:		; preds = %L3Exit, %Loop1
	br label %Loop3

Loop3:		; preds = %Loop3, %Loop2
	br i1 %cond3, label %Loop3, label %L3Exit

L3Exit:		; preds = %Loop3
	br i1 %cond2, label %Loop2, label %L2Exit

L2Exit:		; preds = %L3Exit
	br i1 %cond1, label %Loop1, label %L1Exit

L1Exit:		; preds = %L2Exit
	ret void
}
; ModuleID = 'constexpr.ll'
	%SAType = type { i32, { [2 x float], i64 } }
	%SType = type { i32, { float, { i8 } }, i64 }
global i64 1		; <i64*>:0 [#uses=0]
global i64 74514		; <i64*>:1 [#uses=0]
@t2 = global i32* @t1		; <i32**> [#uses=0]
@t3 = global i32* @t1		; <i32**> [#uses=2]
@t1 = global i32 4		; <i32*> [#uses=2]
@t4 = global i32** @t3		; <i32***> [#uses=1]
@t5 = global i32** @t3		; <i32***> [#uses=0]
@t6 = global i32*** @t4		; <i32****> [#uses=0]
@t7 = global float* inttoptr (i32 12345678 to float*)		; <float**> [#uses=0]
@t9 = global i32 8		; <i32*> [#uses=0]
global i32* bitcast (float* @4 to i32*)		; <i32**>:2 [#uses=0]
global float* @4		; <float**>:3 [#uses=0]
global float 0.000000e+00		; <float*>:4 [#uses=2]
@array = constant [2 x i32] [ i32 12, i32 52 ]		; <[2 x i32]*> [#uses=1]
@arrayPtr = global i32* getelementptr ([2 x i32]* @array, i64 0, i64 0)		; <i32**> [#uses=1]
@arrayPtr5 = global i32** getelementptr (i32** @arrayPtr, i64 5)		; <i32***> [#uses=0]
@somestr = constant [11 x i8] c"hello world"		; <[11 x i8]*> [#uses=2]
@char5 = global i8* getelementptr ([11 x i8]* @somestr, i64 0, i64 5)		; <i8**> [#uses=0]
@char8a = global i32* bitcast (i8* getelementptr ([11 x i8]* @somestr, i64 0, i64 8) to i32*)		; <i32**> [#uses=0]
@char8b = global i8* getelementptr ([11 x i8]* @somestr, i64 0, i64 8)		; <i8**> [#uses=0]
@S1 = global %SType* null		; <%SType**> [#uses=1]
@S2c = constant %SType {
    i32 1, 
    { float, { i8 } } { float 2.000000e+00, { i8 } { i8 3 } }, 
    i64 4 }		; <%SType*> [#uses=3]
@S3c = constant %SAType { i32 1, { [2 x float], i64 } { [2 x float] [ float 2.000000e+00, float 3.000000e+00 ], i64 4 } }		; <%SAType*> [#uses=1]
@S1ptr = global %SType** @S1		; <%SType***> [#uses=0]
@S2 = global %SType* @S2c		; <%SType**> [#uses=0]
@S3 = global %SAType* @S3c		; <%SAType**> [#uses=0]
@S1fld1a = global float* getelementptr (%SType* @S2c, i64 0, i32 1, i32 0)		; <float**> [#uses=0]
@S1fld1b = global float* getelementptr (%SType* @S2c, i64 0, i32 1, i32 0)		; <float**> [#uses=1]
@S1fld1bptr = global float** @S1fld1b		; <float***> [#uses=0]
@S2fld3 = global i8* getelementptr (%SType* @S2c, i64 0, i32 1, i32 1, i32 0)		; <i8**> [#uses=0]

; ModuleID = 'constpointer.ll'
@cpt3 = global i32* @cpt1		; <i32**> [#uses=1]
@cpt1 = global i32 4		; <i32*> [#uses=2]
@cpt4 = global i32** @cpt3		; <i32***> [#uses=0]
@cpt2 = global i32* @cpt1		; <i32**> [#uses=0]
global float* @7		; <float**>:0 [#uses=0]
global float* @7		; <float**>:1 [#uses=0]
global float 0.000000e+00		; <float*>:2 [#uses=3]
global float* @7		; <float**>:3 [#uses=0]
@fptr = global void ()* @f		; <void ()**> [#uses=0]
@sptr1 = global [11 x i8]* @somestr		; <[11 x i8]**> [#uses=0]
@somestr2 = constant [11 x i8] c"hello world"		; <[11 x i8]*> [#uses=2]
@sptr2 = global [11 x i8]* @somestr2		; <[11 x i8]**> [#uses=0]

declare void @f()
; ModuleID = 'escaped_label.ll'

define i32 @foo3() {
	br label "foo`~!@#$%^&*()-_=+{}[]\\|;:',<.>/?"

"foo`~!@#$%^&*()-_=+{}[]\\|;:',<.>/?":		; preds = %0
	ret i32 17
}
; ModuleID = 'float.ll'
@F1 = global float 4.000000e+00		; <float*> [#uses=0]
@D1 = global double 4.000000e+00		; <double*> [#uses=0]
; ModuleID = 'fold-fpcast.ll'

define i32 @test1() {
	ret i32 1080872141
}

define float @test1002() {
	ret float 0x36E1000000000000
}

define i64 @test3() {
	ret i64 4614256656431372362
}

define double @test4() {
	ret double 2.075076e-322
}
; ModuleID = 'forwardreftest.ll'
	%myfn = type float (i32, double, i32, i16)
	%myty = type i32
	%thisfuncty = type i32 (i32)*

declare void @F(%thisfuncty, %thisfuncty, %thisfuncty)

define i32 @zarro2(i32 %Func) {
Startup:
	add i32 0, 10		; <i32>:0 [#uses=0]
	ret i32 0
}

define i32 @test1004(i32) {
	call void @F( %thisfuncty @zarro2, %thisfuncty @test1004, %thisfuncty @foozball )
	ret i32 0
}

define i32 @foozball(i32) {
	ret i32 0
}

; ModuleID = 'globalredefinition.ll'
@A = global i32* @B		; <i32**> [#uses=0]
@B = global i32 7		; <i32*> [#uses=1]

define void @X() {
	ret void
}
; ModuleID = 'global_section.ll'
@GlobSec = global i32 4, section "foo", align 16

define void @test1005() section "bar" {
	ret void
}

; ModuleID = 'globalvars.ll'
@MyVar = external global i32		; <i32*> [#uses=1]
@MyIntList = external global { \2*, i32 }		; <{ \2*, i32 }*> [#uses=1]
external global i32		; <i32*>:0 [#uses=0]
@AConst = constant i32 123		; <i32*> [#uses=0]
@AString = constant [4 x i8] c"test"		; <[4 x i8]*> [#uses=0]
@ZeroInit = global { [100 x i32], [40 x float] } zeroinitializer		; <{ [100 x i32], [40 x float] }*> [#uses=0]

define i32 @foo10015(i32 %blah) {
	store i32 5, i32* @MyVar
	%idx = getelementptr { \2*, i32 }* @MyIntList, i64 0, i32 1		; <i32*> [#uses=1]
	store i32 12, i32* %idx
	ret i32 %blah
}
; ModuleID = 'indirectcall2.ll'

define i64 @test1006(i64 %X) {
	ret i64 %X
}

define i64 @fib(i64 %n) {
; <label>:0
	%T = icmp ult i64 %n, 2		; <i1> [#uses=1]
	br i1 %T, label %BaseCase, label %RecurseCase

RecurseCase:		; preds = %0
	%result = call i64 @test1006( i64 %n )		; <i64> [#uses=0]
	br label %BaseCase

BaseCase:		; preds = %RecurseCase, %0
	%X = phi i64 [ 1, %0 ], [ 2, %RecurseCase ]		; <i64> [#uses=1]
	ret i64 %X
}
; ModuleID = 'indirectcall.ll'

declare i32 @atoi(i8*)

define i64 @fibonacc(i64 %n) {
	icmp ult i64 %n, 2		; <i1>:1 [#uses=1]
	br i1 %1, label %BaseCase, label %RecurseCase

BaseCase:		; preds = %0
	ret i64 1

RecurseCase:		; preds = %0
	%n2 = sub i64 %n, 2		; <i64> [#uses=1]
	%n1 = sub i64 %n, 1		; <i64> [#uses=1]
	%f2 = call i64 @fibonacc( i64 %n2 )		; <i64> [#uses=1]
	%f1 = call i64 @fibonacc( i64 %n1 )		; <i64> [#uses=1]
	%result = add i64 %f2, %f1		; <i64> [#uses=1]
	ret i64 %result
}

define i64 @realmain(i32 %argc, i8** %argv) {
; <label>:0
	icmp eq i32 %argc, 2		; <i1>:1 [#uses=1]
	br i1 %1, label %HasArg, label %Continue

HasArg:		; preds = %0
	%n1 = add i32 1, 1		; <i32> [#uses=1]
	br label %Continue

Continue:		; preds = %HasArg, %0
	%n = phi i32 [ %n1, %HasArg ], [ 1, %0 ]		; <i32> [#uses=1]
	%N = sext i32 %n to i64		; <i64> [#uses=1]
	%F = call i64 @fib( i64 %N )		; <i64> [#uses=1]
	ret i64 %F
}

define i64 @trampoline(i64 %n, i64 (i64)* %fibfunc) {
	%F = call i64 %fibfunc( i64 %n )		; <i64> [#uses=1]
	ret i64 %F
}

define i32 @main2() {
	%Result = call i64 @trampoline( i64 10, i64 (i64)* @fib )		; <i64> [#uses=1]
	%Result.upgrd.1 = trunc i64 %Result to i32		; <i32> [#uses=1]
	ret i32 %Result.upgrd.1
}
; ModuleID = 'inlineasm.ll'
module asm "this is an inline asm block"
module asm "this is another inline asm block"

define i32 @test1007() {
	%X = call i32 asm "tricky here $0, $1", "=r,r"( i32 4 )		; <i32> [#uses=1]
	call void asm sideeffect "eieio", ""( )
	ret i32 %X
}
; ModuleID = 'instructions.ll'

define i32 @test_extractelement(<4 x i32> %V) {
	%R = extractelement <4 x i32> %V, i32 1		; <i32> [#uses=1]
	ret i32 %R
}

define <4 x i32> @test_insertelement(<4 x i32> %V) {
	%R = insertelement <4 x i32> %V, i32 0, i32 0		; <<4 x i32>> [#uses=1]
	ret <4 x i32> %R
}

define <4 x i32> @test_shufflevector_u(<4 x i32> %V) {
	%R = shufflevector <4 x i32> %V, <4 x i32> %V, <4 x i32> < i32 1, i32 undef, i32 7, i32 2 >		; <<4 x i32>> [#uses=1]
	ret <4 x i32> %R
}

define <4 x float> @test_shufflevector_f(<4 x float> %V) {
	%R = shufflevector <4 x float> %V, <4 x float> undef, <4 x i32> < i32 1, i32 undef, i32 7, i32 2 >		; <<4 x float>> [#uses=1]
	ret <4 x float> %R
}
; ModuleID = 'intrinsics.ll'

declare i1 @llvm.isunordered.f32(float, float)

declare i1 @llvm.isunordered.f64(double, double)

declare void @llvm.prefetch(i8*, i32, i32)

declare float @llvm.sqrt.f32(float)

declare double @llvm.sqrt.f64(double)

define void @libm() {
	fcmp uno float 1.000000e+00, 2.000000e+00		; <i1>:1 [#uses=0]
	fcmp uno double 3.000000e+00, 4.000000e+00		; <i1>:2 [#uses=0]
	call void @llvm.prefetch( i8* null, i32 1, i32 3 )
	call float @llvm.sqrt.f32( float 5.000000e+00 )		; <float>:3 [#uses=0]
	call double @llvm.sqrt.f64( double 6.000000e+00 )		; <double>:4 [#uses=0]
	call i8 @llvm.ctpop.i8( i8 10 )		; <i32>:5 [#uses=1]
	call i16 @llvm.ctpop.i16( i16 11 )		; <i32>:7 [#uses=1]
	call i32 @llvm.ctpop.i32( i32 12 )		; <i32>:9 [#uses=1]
	call i64 @llvm.ctpop.i64( i64 13 )		; <i32>:11 [#uses=1]
	call i8 @llvm.ctlz.i8( i8 14 )		; <i32>:13 [#uses=1]
	call i16 @llvm.ctlz.i16( i16 15 )		; <i32>:15 [#uses=1]
	call i32 @llvm.ctlz.i32( i32 16 )		; <i32>:17 [#uses=1]
	call i64 @llvm.ctlz.i64( i64 17 )		; <i32>:19 [#uses=1]
	call i8 @llvm.cttz.i8( i8 18 )		; <i32>:21 [#uses=1]
	call i16 @llvm.cttz.i16( i16 19 )		; <i32>:23 [#uses=1]
	call i32 @llvm.cttz.i32( i32 20 )		; <i32>:25 [#uses=1]
	call i64 @llvm.cttz.i64( i64 21 )		; <i32>:27 [#uses=1]
	ret void
}

declare i8 @llvm.ctpop.i8(i8)

declare i16 @llvm.ctpop.i16(i16)

declare i32 @llvm.ctpop.i32(i32)

declare i64 @llvm.ctpop.i64(i64)

declare i8 @llvm.ctlz.i8(i8)

declare i16 @llvm.ctlz.i16(i16)

declare i32 @llvm.ctlz.i32(i32)

declare i64 @llvm.ctlz.i64(i64)

declare i8 @llvm.cttz.i8(i8)

declare i16 @llvm.cttz.i16(i16)

declare i32 @llvm.cttz.i32(i32)

declare i64 @llvm.cttz.i64(i64)

; ModuleID = 'packed.ll'
@foo1 = external global <4 x float>		; <<4 x float>*> [#uses=2]
@foo102 = external global <2 x i32>		; <<2 x i32>*> [#uses=2]

define void @main3() {
	store <4 x float> < float 1.000000e+00, float 2.000000e+00, float 3.000000e+00, float 4.000000e+00 >, <4 x float>* @foo1
	store <2 x i32> < i32 4, i32 4 >, <2 x i32>* @foo102
	%l1 = load <4 x float>* @foo1		; <<4 x float>> [#uses=0]
	%l2 = load <2 x i32>* @foo102		; <<2 x i32>> [#uses=0]
	ret void
}

; ModuleID = 'properties.ll'
target datalayout = "e-p:32:32"
target triple = "proc-vend-sys"
deplibs = [ "m", "c" ]
; ModuleID = 'prototype.ll'

declare i32 @bar1017(i32 %in)

define i32 @foo1016(i32 %blah) {
	%xx = call i32 @bar1017( i32 %blah )		; <i32> [#uses=1]
	ret i32 %xx
}

; ModuleID = 'recursivetype.ll'
	%list = type { %list*, i32 }

declare i8* @malloc(i32)

define void @InsertIntoListTail(%list** %L, i32 %Data) {
bb1:
	%reg116 = load %list** %L		; <%list*> [#uses=1]
	%cast1004 = inttoptr i64 0 to %list*		; <%list*> [#uses=1]
	%cond1000 = icmp eq %list* %reg116, %cast1004		; <i1> [#uses=1]
	br i1 %cond1000, label %bb3, label %bb2

bb2:		; preds = %bb2, %bb1
	%reg117 = phi %list** [ %reg118, %bb2 ], [ %L, %bb1 ]		; <%list**> [#uses=1]
	%cast1010 = bitcast %list** %reg117 to %list***		; <%list***> [#uses=1]
	%reg118 = load %list*** %cast1010		; <%list**> [#uses=3]
	%reg109 = load %list** %reg118		; <%list*> [#uses=1]
	%cast1005 = inttoptr i64 0 to %list*		; <%list*> [#uses=1]
	%cond1001 = icmp ne %list* %reg109, %cast1005		; <i1> [#uses=1]
	br i1 %cond1001, label %bb2, label %bb3

bb3:		; preds = %bb2, %bb1
	%reg119 = phi %list** [ %reg118, %bb2 ], [ %L, %bb1 ]		; <%list**> [#uses=1]
	%cast1006 = bitcast %list** %reg119 to i8**		; <i8**> [#uses=1]
	%reg111 = call i8* @malloc( i32 16 )		; <i8*> [#uses=3]
	store i8* %reg111, i8** %cast1006
	%reg111.upgrd.1 = ptrtoint i8* %reg111 to i64		; <i64> [#uses=1]
	%reg1002 = add i64 %reg111.upgrd.1, 8		; <i64> [#uses=1]
	%reg1002.upgrd.2 = inttoptr i64 %reg1002 to i8*		; <i8*> [#uses=1]
	%cast1008 = bitcast i8* %reg1002.upgrd.2 to i32*		; <i32*> [#uses=1]
	store i32 %Data, i32* %cast1008
	%cast1003 = inttoptr i64 0 to i64*		; <i64*> [#uses=1]
	%cast1009 = bitcast i8* %reg111 to i64**		; <i64**> [#uses=1]
	store i64* %cast1003, i64** %cast1009
	ret void
}

define %list* @FindData(%list* %L, i32 %Data) {
bb1:
	br label %bb2

bb2:		; preds = %bb6, %bb1
	%reg115 = phi %list* [ %reg116, %bb6 ], [ %L, %bb1 ]		; <%list*> [#uses=4]
	%cast1014 = inttoptr i64 0 to %list*		; <%list*> [#uses=1]
	%cond1011 = icmp ne %list* %reg115, %cast1014		; <i1> [#uses=1]
	br i1 %cond1011, label %bb4, label %bb3

bb3:		; preds = %bb2
	ret %list* null

bb4:		; preds = %bb2
	%idx = getelementptr %list* %reg115, i64 0, i32 1		; <i32*> [#uses=1]
	%reg111 = load i32* %idx		; <i32> [#uses=1]
	%cond1013 = icmp ne i32 %reg111, %Data		; <i1> [#uses=1]
	br i1 %cond1013, label %bb6, label %bb5

bb5:		; preds = %bb4
	ret %list* %reg115

bb6:		; preds = %bb4
	%idx2 = getelementptr %list* %reg115, i64 0, i32 0		; <%list**> [#uses=1]
	%reg116 = load %list** %idx2		; <%list*> [#uses=1]
	br label %bb2
}
; ModuleID = 'simplecalltest.ll'
	%FunTy = type i32 (i32)

define void @invoke1019(%FunTy* %x) {
	%foo = call i32 %x( i32 123 )		; <i32> [#uses=0]
	ret void
}

define i32 @main4(i32 %argc, i8** %argv, i8** %envp) {
	%retval = call i32 @test1008( i32 %argc )		; <i32> [#uses=2]
	%two = add i32 %retval, %retval		; <i32> [#uses=1]
	%retval2 = call i32 @test1008( i32 %argc )		; <i32> [#uses=1]
	%two2 = add i32 %two, %retval2		; <i32> [#uses=1]
	call void @invoke1019( %FunTy* @test1008 )
	ret i32 %two2
}

define i32 @test1008(i32 %i0) {
	ret i32 %i0
}
; ModuleID = 'smallest.ll'
; ModuleID = 'small.ll'
	%x = type i32

define i32 @foo1020(i32 %in) {
label:
	ret i32 2
}
; ModuleID = 'testalloca.ll'
	%inners = type { float, { i8 } }
	%struct = type { i32, %inners, i64 }

define i32 @testfunction(i32 %i0, i32 %j0) {
	alloca i8, i32 5		; <i8*>:1 [#uses=0]
	%ptr = alloca i32		; <i32*> [#uses=2]
	store i32 3, i32* %ptr
	%val = load i32* %ptr		; <i32> [#uses=0]
	%sptr = alloca %struct		; <%struct*> [#uses=2]
	%nsptr = getelementptr %struct* %sptr, i64 0, i32 1		; <%inners*> [#uses=1]
	%ubsptr = getelementptr %inners* %nsptr, i64 0, i32 1		; <{ i8 }*> [#uses=1]
	%idx = getelementptr { i8 }* %ubsptr, i64 0, i32 0		; <i8*> [#uses=1]
	store i8 4, i8* %idx
	%fptr = getelementptr %struct* %sptr, i64 0, i32 1, i32 0		; <float*> [#uses=1]
	store float 4.000000e+00, float* %fptr
	ret i32 3
}
; ModuleID = 'testconstants.ll'
@somestr3 = constant [11 x i8] c"hello world"
@array99 = constant [2 x i32] [ i32 12, i32 52 ]
constant { i32, i32 } { i32 4, i32 3 }		; <{ i32, i32 }*>:0 [#uses=0]

define [2 x i32]* @testfunction99(i32 %i0, i32 %j0) {
	ret [2 x i32]* @array
}

define i8* @otherfunc(i32, double) {
	%somestr = getelementptr [11 x i8]* @somestr3, i64 0, i64 0		; <i8*> [#uses=1]
	ret i8* %somestr
}

define i8* @yetanotherfunc(i32, double) {
	ret i8* null
}

define i32 @negativeUnsigned() {
	ret i32 -1
}

define i32 @largeSigned() {
	ret i32 -394967296
}
; ModuleID = 'testlogical.ll'

define i32 @simpleAdd(i32 %i0, i32 %j0) {
	%t1 = xor i32 %i0, %j0		; <i32> [#uses=1]
	%t2 = or i32 %i0, %j0		; <i32> [#uses=1]
	%t3 = and i32 %t1, %t2		; <i32> [#uses=1]
	ret i32 %t3
}
; ModuleID = 'testmemory.ll'
	%complexty = type { i32, { [4 x i8*], float }, double }
	%struct = type { i32, { float, { i8 } }, i64 }

define i32 @main6() {
	call i32 @testfunction98( i64 0, i64 1 )
	ret i32 0
}

define i32 @testfunction98(i64 %i0, i64 %j0) {
	%array0 = malloc [4 x i8]		; <[4 x i8]*> [#uses=2]
	%size = add i32 2, 2		; <i32> [#uses=1]
	%array1 = malloc i8, i32 4		; <i8*> [#uses=1]
	%array2 = malloc i8, i32 %size		; <i8*> [#uses=1]
	%idx = getelementptr [4 x i8]* %array0, i64 0, i64 2		; <i8*> [#uses=1]
	store i8 123, i8* %idx
	free [4 x i8]* %array0
	free i8* %array1
	free i8* %array2
	%aa = alloca %complexty, i32 5		; <%complexty*> [#uses=1]
	%idx2 = getelementptr %complexty* %aa, i64 %i0, i32 1, i32 0, i64 %j0		; <i8**> [#uses=1]
	store i8* null, i8** %idx2
	%ptr = alloca i32		; <i32*> [#uses=2]
	store i32 3, i32* %ptr
	%val = load i32* %ptr		; <i32> [#uses=0]
	%sptr = alloca %struct		; <%struct*> [#uses=1]
	%ubsptr = getelementptr %struct* %sptr, i64 0, i32 1, i32 1		; <{ i8 }*> [#uses=1]
	%idx3 = getelementptr { i8 }* %ubsptr, i64 0, i32 0		; <i8*> [#uses=1]
	store i8 4, i8* %idx3
	ret i32 3
}
; ModuleID = 'testswitch.ll'
	%int = type i32

define i32 @squared(i32 %i0) {
	switch i32 %i0, label %Default [
		 i32 1, label %Case1
		 i32 2, label %Case2
		 i32 4, label %Case4
	]

Default:		; preds = %0
	ret i32 -1

Case1:		; preds = %0
	ret i32 1

Case2:		; preds = %0
	ret i32 4

Case4:		; preds = %0
	ret i32 16
}
; ModuleID = 'testvarargs.ll'

declare i32 @printf(i8*, ...)

define i32 @testvarar() {
	call i32 (i8*, ...)* @printf( i8* null, i32 12, i8 42 )		; <i32>:1 [#uses=1]
	ret i32 %1
}
; ModuleID = 'undefined.ll'
@X2 = global i32 undef		; <i32*> [#uses=0]

declare i32 @atoi(i8*)

define i32 @test1009() {
	ret i32 undef
}

define i32 @test1003() {
	%X = add i32 undef, 1		; <i32> [#uses=1]
	ret i32 %X
}
; ModuleID = 'unreachable.ll'

declare void @bar()

define i32 @foo1021() {
	unreachable
}

define double @xyz() {
	call void @bar( )
	unreachable
}
; ModuleID = 'varargs.ll'

declare void @llvm.va_start(i8* %ap)

declare void @llvm.va_copy(i8* %aq, i8* %ap)

declare void @llvm.va_end(i8* %ap)

define i32 @test1010(i32 %X, ...) {
	%ap = alloca i8*		; <i8**> [#uses=4]
	%va.upgrd.1 = bitcast i8** %ap to i8*		; <i8*> [#uses=1]
	call void @llvm.va_start( i8* %va.upgrd.1 )
	%tmp = va_arg i8** %ap, i32		; <i32> [#uses=1]
	%aq = alloca i8*		; <i8**> [#uses=2]
	%va0.upgrd.2 = bitcast i8** %aq to i8*		; <i8*> [#uses=1]
	%va1.upgrd.3 = bitcast i8** %ap to i8*		; <i8*> [#uses=1]
	call void @llvm.va_copy( i8* %va0.upgrd.2, i8* %va1.upgrd.3 )
	%va.upgrd.4 = bitcast i8** %aq to i8*		; <i8*> [#uses=1]
	call void @llvm.va_end( i8* %va.upgrd.4 )
	%va.upgrd.5 = bitcast i8** %ap to i8*		; <i8*> [#uses=1]
	call void @llvm.va_end( i8* %va.upgrd.5 )
	ret i32 %tmp
}
; ModuleID = 'varargs_new.ll'

declare void @llvm.va_start(i8*)

declare void @llvm.va_copy(i8*, i8*)

declare void @llvm.va_end(i8*)

define i32 @test1011(i32 %X, ...) {
	%ap = alloca i8*		; <i8**> [#uses=4]
	%aq = alloca i8*		; <i8**> [#uses=2]
	%va.upgrd.1 = bitcast i8** %ap to i8*		; <i8*> [#uses=1]
	call void @llvm.va_start( i8* %va.upgrd.1 )
	%tmp = va_arg i8** %ap, i32		; <i32> [#uses=1]
	%apv = load i8** %ap		; <i8*> [#uses=1]
	%va0.upgrd.2 = bitcast i8** %aq to i8*		; <i8*> [#uses=1]
	%va1.upgrd.3 = bitcast i8* %apv to i8*		; <i8*> [#uses=1]
	call void @llvm.va_copy( i8* %va0.upgrd.2, i8* %va1.upgrd.3 )
	%va.upgrd.4 = bitcast i8** %aq to i8*		; <i8*> [#uses=1]
	call void @llvm.va_end( i8* %va.upgrd.4 )
	%va.upgrd.5 = bitcast i8** %ap to i8*		; <i8*> [#uses=1]
	call void @llvm.va_end( i8* %va.upgrd.5 )
	ret i32 %tmp
}
; ModuleID = 'weirdnames.ll'
	"&^ " = type { i32 }
@"%.*+ foo" = global "&^ " { i32 5 }		; <"&^ "*> [#uses=0]
@"0" = global float 0.000000e+00		; <float*> [#uses=0]
