; RUN: opt -S < %s -mtriple=unknown -instcombine -instcombine-infinite-loop-threshold=2 | FileCheck -check-prefixes=CHECK,CHECK32 %s
; RUN: opt -S < %s -mtriple=msp430 -instcombine -instcombine-infinite-loop-threshold=2 | FileCheck -check-prefixes=CHECK,CHECK16 %s
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-f80:128:128-v64:64:64-v128:128:128-a0:0:64-f80:32:32-n8:16:32-S32"

@G = constant [3 x i8] c"%s\00"		; <[3 x i8]*> [#uses=1]

declare i32 @sprintf(i8*, i8*, ...)

define void @foo(i8* %P, i32* %X) {
	call i32 (i8*, i8*, ...) @sprintf( i8* %P, i8* getelementptr ([3 x i8], [3 x i8]* @G, i32 0, i32 0), i32* %X )		; <i32>:1 [#uses=0]
	ret void
}

; PR1307
@str = internal constant [5 x i8] c"foog\00"
@str1 = internal constant [8 x i8] c"blahhh!\00"
@str2 = internal constant [5 x i8] c"Ponk\00"

define i8* @test1() {
        %tmp3 = tail call i8* @strchr( i8* getelementptr ([5 x i8], [5 x i8]* @str, i32 0, i32 2), i32 103 )              ; <i8*> [#uses=1]
        ret i8* %tmp3

; CHECK-LABEL: @test1(
; CHECK: ret i8* getelementptr inbounds ([5 x i8], [5 x i8]* @str, i32 0, i32 3)
}

declare i8* @strchr(i8*, i32)

define i8* @test2() {
        %tmp3 = tail call i8* @strchr( i8* getelementptr ([8 x i8], [8 x i8]* @str1, i32 0, i32 2), i32 0 )               ; <i8*> [#uses=1]
        ret i8* %tmp3

; CHECK-LABEL: @test2(
; CHECK: ret i8* getelementptr inbounds ([8 x i8], [8 x i8]* @str1, i32 0, i32 7)
}

define i8* @test3() {
entry:
        %tmp3 = tail call i8* @strchr( i8* getelementptr ([5 x i8], [5 x i8]* @str2, i32 0, i32 1), i32 80 )              ; <i8*> [#uses=1]
        ret i8* %tmp3

; CHECK-LABEL: @test3(
; CHECK: ret i8* null
}

@_2E_str = external constant [5 x i8]		; <[5 x i8]*> [#uses=1]

declare i32 @memcmp(i8*, i8*, i32) nounwind readonly

define i1 @PR2341(i8** %start_addr) {
entry:
	%tmp4 = load i8*, i8** %start_addr, align 4		; <i8*> [#uses=1]
	%tmp5 = call i32 @memcmp( i8* %tmp4, i8* getelementptr ([5 x i8], [5 x i8]* @_2E_str, i32 0, i32 0), i32 4 ) nounwind readonly 		; <i32> [#uses=1]
	%tmp6 = icmp eq i32 %tmp5, 0		; <i1> [#uses=1]
	ret i1 %tmp6

; CHECK-LABEL: @PR2341(
; CHECK: i32
}

define i32 @PR4284() nounwind {
entry:
	%c0 = alloca i8, align 1		; <i8*> [#uses=2]
	%c2 = alloca i8, align 1		; <i8*> [#uses=2]
	store i8 64, i8* %c0
	store i8 -127, i8* %c2
	%call = call i32 @memcmp(i8* %c0, i8* %c2, i32 1)		; <i32> [#uses=1]
	ret i32 %call

; CHECK-LABEL: @PR4284(
; CHECK: ret i32 -65
}

%struct.__sFILE = type { i8*, i32, i32, i16, i16, %struct.__sbuf, i32, i8*, i32 (i8*)*, i32 (i8*, i8*, i32)*, i64 (i8*, i64, i32)*, i32 (i8*, i8*, i32)*, %struct.__sbuf, i8*, i32, [3 x i8], [1 x i8], %struct.__sbuf, i32, i64, %struct.pthread_mutex*, %struct.pthread*, i32, i32, %union.anon }
%struct.__sbuf = type { i8*, i32, [4 x i8] }
%struct.pthread = type opaque
%struct.pthread_mutex = type opaque
%union.anon = type { i64, [120 x i8] }
@.str13 = external constant [2 x i8]		; <[2 x i8]*> [#uses=1]
@.str14 = external constant [2 x i8]		; <[2 x i8]*> [#uses=1]

define i32 @PR4641(i32 %argc, i8** %argv) nounwind {
entry:
	call void @exit(i32 0) nounwind
	%cond392 = select i1 undef, i8* getelementptr ([2 x i8], [2 x i8]* @.str13, i32 0, i32 0), i8* getelementptr ([2 x i8], [2 x i8]* @.str14, i32 0, i32 0)		; <i8*> [#uses=1]
	%call393 = call %struct.__sFILE* @fopen(i8* undef, i8* %cond392) nounwind		; <%struct.__sFILE*> [#uses=0]
	unreachable
}

declare %struct.__sFILE* @fopen(i8*, i8*)

declare void @exit(i32)

define i32 @PR4645() {
entry:
	br label %if.then

lor.lhs.false:		; preds = %while.body
	br i1 undef, label %if.then, label %for.cond

if.then:		; preds = %lor.lhs.false, %while.body
	call void @exit(i32 1)
	br label %for.cond

for.cond:		; preds = %for.end, %if.then, %lor.lhs.false
	%j.0 = phi i32 [ %inc47, %for.end ], [ 0, %if.then ], [ 0, %lor.lhs.false ]		; <i32> [#uses=1]
	unreachable

for.end:		; preds = %for.cond20
	%inc47 = add i32 %j.0, 1		; <i32> [#uses=1]
	br label %for.cond
}

@h = constant [2 x i8] c"h\00"		; <[2 x i8]*> [#uses=1]
@hel = constant [4 x i8] c"hel\00"		; <[4 x i8]*> [#uses=1]
@hello_u = constant [8 x i8] c"hello_u\00"		; <[8 x i8]*> [#uses=1]

define i32 @MemCpy() {
  %h_p = getelementptr [2 x i8], [2 x i8]* @h, i32 0, i32 0
  %hel_p = getelementptr [4 x i8], [4 x i8]* @hel, i32 0, i32 0
  %hello_u_p = getelementptr [8 x i8], [8 x i8]* @hello_u, i32 0, i32 0
  %target = alloca [1024 x i8]
  %target_p = getelementptr [1024 x i8], [1024 x i8]* %target, i32 0, i32 0
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* align 2 %target_p, i8* align 2 %h_p, i32 2, i1 false)
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* align 4 %target_p, i8* align 4 %hel_p, i32 4, i1 false)
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* align 8 %target_p, i8* align 8 %hello_u_p, i32 8, i1 false)
  ret i32 0

; CHECK-LABEL: @MemCpy(
; CHECK-NOT: llvm.memcpy
; CHECK: ret i32 0
}

declare void @llvm.memcpy.p0i8.p0i8.i32(i8* nocapture, i8* nocapture, i32, i1) nounwind

declare i32 @strcmp(i8*, i8*) #0

define void @test9(i8* %x) {
; CHECK-LABEL: @test9(
; CHECK-NOT: strcmp
  %y = call i32 @strcmp(i8* %x, i8* %x) #1
  ret void
}

; PR30484 - https://llvm.org/bugs/show_bug.cgi?id=30484
; These aren't the library functions you're looking for...

declare i32 @isdigit(i8)
declare i32 @isascii(i8)
declare i32 @toascii(i8)

define i32 @fake_isdigit(i8 %x) {
; CHECK-LABEL: @fake_isdigit(
; CHECK-NEXT:    [[Y:%.*]] = call i32 @isdigit(i8 %x)
; CHECK-NEXT:    ret i32 [[Y]]
;
  %y = call i32 @isdigit(i8 %x)
  ret i32 %y
}

define i32 @fake_isascii(i8 %x) {
; CHECK-LABEL: @fake_isascii(
; CHECK-NEXT:    [[Y:%.*]] = call i32 @isascii(i8 %x)
; CHECK-NEXT:    ret i32 [[Y]]
;
  %y = call i32 @isascii(i8 %x)
  ret i32 %y
}

define i32 @fake_toascii(i8 %x) {
; CHECK-LABEL: @fake_toascii(
; CHECK-NEXT:    [[Y:%.*]] = call i32 @toascii(i8 %x)
; CHECK-NEXT:    ret i32 [[Y]]
;
  %y = call i32 @toascii(i8 %x)
  ret i32 %y
}

declare double @pow(double, double)
declare double @exp2(double)

; check to make sure only the correct libcall attributes are used
define double @fake_exp2(double %x) {
; CHECK-LABEL: @fake_exp2(
; CHECK-NEXT:    [[Y:%.*]] = call double @exp2(double %x)
; CHECK-NEXT:    ret double [[Y]]

  %y = call inreg double @pow(double inreg 2.0, double inreg %x)
  ret double %y
}
define double @fake_ldexp(i32 %x) {
; CHECK32-LABEL: @fake_ldexp(
; CHECK32-NEXT:    [[Z:%.*]] = call double @ldexp(double 1.0{{.*}}, i32 %x)
; CHECK32-NEXT:    ret double [[Z]]

; CHECK16-LABEL: @fake_ldexp(
; CHECK16-NEXT:    [[Y:%.*]] = sitofp i32 %x to double
; CHECK16-NEXT:    [[Z:%.*]] = call inreg double @exp2(double [[Y]])
; CHECK16-NEXT:    ret double [[Z]]

  %y = sitofp i32 %x to double
  %z = call inreg double @exp2(double %y)
  ret double %z
}
define double @fake_ldexp_16(i16 %x) {
; CHECK32-LABEL: @fake_ldexp_16(
; CHECK32-NEXT:    [[Y:%.*]] = sext i16 %x to i32
; CHECK32-NEXT:    [[Z:%.*]] = call double @ldexp(double 1.0{{.*}}, i32 [[Y]])
; CHECK32-NEXT:    ret double [[Z]]

; CHECK16-LABEL: @fake_ldexp_16(
; CHECK16-NEXT:    [[Z:%.*]] = call double @ldexp(double 1.0{{.*}}, i16 %x)
; CHECK16-NEXT:    ret double [[Z]]

  %y = sitofp i16 %x to double
  %z = call inreg double @exp2(double %y)
  ret double %z
}

; PR50885 - this would crash in ValueTracking.

declare i32 @snprintf(i8*, double, i32*)

define i32 @fake_snprintf(i32 %buf, double %len, i32 * %str) {
; CHECK-LABEL: @fake_snprintf(
; CHECK-NEXT:    [[CALL:%.*]] = call i32 @snprintf(i8* undef, double [[LEN:%.*]], i32* [[STR:%.*]])
; CHECK-NEXT:    ret i32 [[CALL]]
;
  %call = call i32 @snprintf(i8* undef, double %len, i32* %str)
  ret i32 %call
}

attributes #0 = { nobuiltin }
attributes #1 = { builtin }
