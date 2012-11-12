; Test a pile of objectsize bounds checking.
; RUN: opt < %s -instcombine -S | FileCheck %s
; We need target data to get the sizes of the arrays and structures.
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"

@a = private global [60 x i8] zeroinitializer, align 1 ; <[60 x i8]*>
@.str = private constant [8 x i8] c"abcdefg\00"   ; <[8 x i8]*>

define i32 @foo() nounwind {
; CHECK: @foo
; CHECK-NEXT: ret i32 60
  %1 = call i32 @llvm.objectsize.i32(i8* getelementptr inbounds ([60 x i8]* @a, i32 0, i32 0), i1 false)
  ret i32 %1
}

define i8* @bar() nounwind {
; CHECK: @bar
entry:
  %retval = alloca i8*
  %0 = call i32 @llvm.objectsize.i32(i8* getelementptr inbounds ([60 x i8]* @a, i32 0, i32 0), i1 false)
  %cmp = icmp ne i32 %0, -1
; CHECK: br i1 true
  br i1 %cmp, label %cond.true, label %cond.false

cond.true:
  %1 = load i8** %retval
  ret i8* %1

cond.false:
  %2 = load i8** %retval
  ret i8* %2
}

define i32 @f() nounwind {
; CHECK: @f
; CHECK-NEXT: ret i32 0
  %1 = call i32 @llvm.objectsize.i32(i8* getelementptr ([60 x i8]* @a, i32 1, i32 0), i1 false)
  ret i32 %1
}

@window = external global [0 x i8]

define i1 @baz() nounwind {
; CHECK: @baz
; CHECK-NEXT: objectsize
  %1 = tail call i32 @llvm.objectsize.i32(i8* getelementptr inbounds ([0 x i8]* @window, i32 0, i32 0), i1 false)
  %2 = icmp eq i32 %1, -1
  ret i1 %2
}

define void @test1(i8* %q, i32 %x) nounwind noinline {
; CHECK: @test1
; CHECK: objectsize.i32
entry:
  %0 = call i32 @llvm.objectsize.i32(i8* getelementptr inbounds ([0 x i8]* @window, i32 0, i32 10), i1 false) ; <i64> [#uses=1]
  %1 = icmp eq i32 %0, -1                         ; <i1> [#uses=1]
  br i1 %1, label %"47", label %"46"

"46":                                             ; preds = %entry
  unreachable

"47":                                             ; preds = %entry
  unreachable
}

@.str5 = private constant [9 x i32] [i32 97, i32 98, i32 99, i32 100, i32 0, i32
 101, i32 102, i32 103, i32 0], align 4
define i32 @test2() nounwind {
; CHECK: @test2
; CHECK-NEXT: ret i32 34
  %1 = call i32 @llvm.objectsize.i32(i8* getelementptr (i8* bitcast ([9 x i32]* @.str5 to i8*), i32 2), i1 false)
  ret i32 %1
}

; rdar://7674946
@array = internal global [480 x float] zeroinitializer ; <[480 x float]*> [#uses=1]

declare i8* @__memcpy_chk(i8*, i8*, i32, i32) nounwind

declare i32 @llvm.objectsize.i32(i8*, i1) nounwind readonly

declare i8* @__inline_memcpy_chk(i8*, i8*, i32) nounwind inlinehint

define void @test3() nounwind {
; CHECK: @test3
entry:
  br i1 undef, label %bb11, label %bb12

bb11:
  %0 = getelementptr inbounds float* getelementptr inbounds ([480 x float]* @array, i32 0, i32 128), i32 -127 ; <float*> [#uses=1]
  %1 = bitcast float* %0 to i8*                   ; <i8*> [#uses=1]
  %2 = call i32 @llvm.objectsize.i32(i8* %1, i1 false) ; <i32> [#uses=1]
  %3 = call i8* @__memcpy_chk(i8* undef, i8* undef, i32 512, i32 %2) nounwind ; <i8*> [#uses=0]
; CHECK: unreachable
  unreachable

bb12:
  %4 = getelementptr inbounds float* getelementptr inbounds ([480 x float]* @array, i32 0, i32 128), i32 -127 ; <float*> [#uses=1]
  %5 = bitcast float* %4 to i8*                   ; <i8*> [#uses=1]
  %6 = call i8* @__inline_memcpy_chk(i8* %5, i8* undef, i32 512) nounwind inlinehint ; <i8*> [#uses=0]
; CHECK: @__inline_memcpy_chk
  unreachable
}

; rdar://7718857

%struct.data = type { [100 x i32], [100 x i32], [1024 x i8] }

define i32 @test4(i8** %esc) nounwind ssp {
; CHECK: @test4
entry:
  %0 = alloca %struct.data, align 8
  %1 = bitcast %struct.data* %0 to i8*
  %2 = call i32 @llvm.objectsize.i32(i8* %1, i1 false) nounwind
; CHECK-NOT: @llvm.objectsize
; CHECK: @llvm.memset.p0i8.i32(i8* %1, i8 0, i32 1824, i32 8, i1 false)
  %3 = call i8* @__memset_chk(i8* %1, i32 0, i32 1824, i32 %2) nounwind
  store i8* %1, i8** %esc
  ret i32 0
}

; rdar://7782496
@s = external global i8*

define i8* @test5(i32 %n) nounwind ssp {
; CHECK: @test5
entry:
  %0 = tail call noalias i8* @malloc(i32 20) nounwind
  %1 = tail call i32 @llvm.objectsize.i32(i8* %0, i1 false)
  %2 = load i8** @s, align 8
; CHECK-NOT: @llvm.objectsize
; CHECK: @llvm.memcpy.p0i8.p0i8.i32(i8* %0, i8* %1, i32 10, i32 1, i1 false)
  %3 = tail call i8* @__memcpy_chk(i8* %0, i8* %2, i32 10, i32 %1) nounwind
  ret i8* %0
}

define void @test6(i32 %n) nounwind ssp {
; CHECK: @test6
entry:
  %0 = tail call noalias i8* @malloc(i32 20) nounwind
  %1 = tail call i32 @llvm.objectsize.i32(i8* %0, i1 false)
  %2 = load i8** @s, align 8
; CHECK-NOT: @llvm.objectsize
; CHECK: @__memcpy_chk(i8* %0, i8* %1, i32 30, i32 20)
  %3 = tail call i8* @__memcpy_chk(i8* %0, i8* %2, i32 30, i32 %1) nounwind
  ret void
}

declare i8* @__memset_chk(i8*, i32, i32, i32) nounwind

declare noalias i8* @malloc(i32) nounwind

define i32 @test7(i8** %esc) {
; CHECK: @test7
  %alloc = call noalias i8* @malloc(i32 48) nounwind
  store i8* %alloc, i8** %esc
  %gep = getelementptr inbounds i8* %alloc, i32 16
  %objsize = call i32 @llvm.objectsize.i32(i8* %gep, i1 false) nounwind readonly
; CHECK: ret i32 32
  ret i32 %objsize
}

declare noalias i8* @calloc(i32, i32) nounwind

define i32 @test8(i8** %esc) {
; CHECK: @test8
  %alloc = call noalias i8* @calloc(i32 5, i32 7) nounwind
  store i8* %alloc, i8** %esc
  %gep = getelementptr inbounds i8* %alloc, i32 5
  %objsize = call i32 @llvm.objectsize.i32(i8* %gep, i1 false) nounwind readonly
; CHECK: ret i32 30
  ret i32 %objsize
}

declare noalias i8* @strdup(i8* nocapture) nounwind
declare noalias i8* @strndup(i8* nocapture, i32) nounwind

; CHECK: @test9
define i32 @test9(i8** %esc) {
  %call = tail call i8* @strdup(i8* getelementptr inbounds ([8 x i8]* @.str, i64 0, i64 0)) nounwind
  store i8* %call, i8** %esc, align 8
  %1 = tail call i32 @llvm.objectsize.i32(i8* %call, i1 true)
; CHECK: ret i32 8
  ret i32 %1
}

; CHECK: @test10
define i32 @test10(i8** %esc) {
  %call = tail call i8* @strndup(i8* getelementptr inbounds ([8 x i8]* @.str, i64 0, i64 0), i32 3) nounwind
  store i8* %call, i8** %esc, align 8
  %1 = tail call i32 @llvm.objectsize.i32(i8* %call, i1 true)
; CHECK: ret i32 4
  ret i32 %1
}

; CHECK: @test11
define i32 @test11(i8** %esc) {
  %call = tail call i8* @strndup(i8* getelementptr inbounds ([8 x i8]* @.str, i64 0, i64 0), i32 7) nounwind
  store i8* %call, i8** %esc, align 8
  %1 = tail call i32 @llvm.objectsize.i32(i8* %call, i1 true)
; CHECK: ret i32 8
  ret i32 %1
}

; CHECK: @test12
define i32 @test12(i8** %esc) {
  %call = tail call i8* @strndup(i8* getelementptr inbounds ([8 x i8]* @.str, i64 0, i64 0), i32 8) nounwind
  store i8* %call, i8** %esc, align 8
  %1 = tail call i32 @llvm.objectsize.i32(i8* %call, i1 true)
; CHECK: ret i32 8
  ret i32 %1
}

; CHECK: @test13
define i32 @test13(i8** %esc) {
  %call = tail call i8* @strndup(i8* getelementptr inbounds ([8 x i8]* @.str, i64 0, i64 0), i32 57) nounwind
  store i8* %call, i8** %esc, align 8
  %1 = tail call i32 @llvm.objectsize.i32(i8* %call, i1 true)
; CHECK: ret i32 8
  ret i32 %1
}

; CHECK: @PR13390
define i32 @PR13390(i1 %bool, i8* %a) {
entry:
  %cond = or i1 %bool, true
  br i1 %cond, label %return, label %xpto

xpto:
  %select = select i1 %bool, i8* %select, i8* %a
  %select2 = select i1 %bool, i8* %a, i8* %select2
  %0 = tail call i32 @llvm.objectsize.i32(i8* %select, i1 true)
  %1 = tail call i32 @llvm.objectsize.i32(i8* %select2, i1 true)
  %2 = add i32 %0, %1
; CHECK: ret i32 undef
  ret i32 %2

return:
  ret i32 42
}

; CHECK: @PR13621
define i32 @PR13621(i1 %bool) nounwind {
entry:
  %cond = or i1 %bool, true
  br i1 %cond, label %return, label %xpto

; technically reachable, but this malformed IR may appear as a result of constant propagation
xpto:
  %gep2 = getelementptr i8* %gep, i32 1
  %gep = getelementptr i8* %gep2, i32 1
  %o = call i32 @llvm.objectsize.i32(i8* %gep, i1 true)
; CHECK: ret i32 undef
  ret i32 %o

return:
  ret i32 7
}
