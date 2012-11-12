; RUN: llvm-as %s -o %t1.bc
; RUN: llvm-as < %p/2004-05-07-TypeResolution2.ll -o %t2.bc
; RUN: llvm-link -o %t3.bc %t1.bc %t2.bc

target datalayout = "e-p:32:32"
	%myint = type opaque
	%struct1 = type { i32, void (%struct2*)*, %myint*, i32 (i32*)* }
	%struct2 = type { %struct1 }
@driver1 = global %struct1 zeroinitializer		; <%struct1*> [#uses=1]
@m1 = external global [1 x i8]*		; <[1 x i8]**> [#uses=0]
@str1 = constant [1 x i8] zeroinitializer		; <[1 x i8]*> [#uses=0]
@str2 = constant [2 x i8] zeroinitializer		; <[2 x i8]*> [#uses=0]
@str3 = constant [3 x i8] zeroinitializer		; <[3 x i8]*> [#uses=0]
@str4 = constant [4 x i8] zeroinitializer		; <[4 x i8]*> [#uses=0]
@str5 = constant [5 x i8] zeroinitializer		; <[5 x i8]*> [#uses=0]
@str6 = constant [6 x i8] zeroinitializer		; <[6 x i8]*> [#uses=0]
@str7 = constant [7 x i8] zeroinitializer		; <[7 x i8]*> [#uses=0]
@str8 = constant [8 x i8] zeroinitializer		; <[8 x i8]*> [#uses=0]
@str9 = constant [9 x i8] zeroinitializer		; <[9 x i8]*> [#uses=0]
@stra = constant [10 x i8] zeroinitializer		; <[10 x i8]*> [#uses=0]
@strb = constant [11 x i8] zeroinitializer		; <[11 x i8]*> [#uses=0]
@strc = constant [12 x i8] zeroinitializer		; <[12 x i8]*> [#uses=0]
@strd = constant [13 x i8] zeroinitializer		; <[13 x i8]*> [#uses=0]
@stre = constant [14 x i8] zeroinitializer		; <[14 x i8]*> [#uses=0]
@strf = constant [15 x i8] zeroinitializer		; <[15 x i8]*> [#uses=0]
@strg = constant [16 x i8] zeroinitializer		; <[16 x i8]*> [#uses=0]
@strh = constant [17 x i8] zeroinitializer		; <[17 x i8]*> [#uses=0]

declare void @func(%struct2*)

define void @tty_init() {
entry:
	store volatile void (%struct2*)* @func, void (%struct2*)** getelementptr (%struct1* @driver1, i64 0, i32 1)
	ret void
}
