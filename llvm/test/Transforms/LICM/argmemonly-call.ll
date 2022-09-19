; RUN: opt -aa-pipeline=basic-aa -passes='require<aa>,require<targetir>,require<scalar-evolution>,require<opt-remark-emit>,loop-mssa(licm)' < %s -S | FileCheck %s

declare i32 @foo() readonly argmemonly nounwind
declare i32 @foo2() readonly nounwind
declare i32 @bar(i32* %loc2) readonly argmemonly nounwind

define void @test(i32* %loc) {
; CHECK-LABEL: @test
; CHECK: @foo
; CHECK-LABEL: loop:
  br label %loop

loop:
  %res = call i32 @foo()
  store i32 %res, i32* %loc
  br label %loop
}

; Negative test: show argmemonly is required
define void @test_neg(i32* %loc) {
; CHECK-LABEL: @test_neg
; CHECK-LABEL: loop:
; CHECK: @foo
  br label %loop

loop:
  %res = call i32 @foo2()
  store i32 %res, i32* %loc
  br label %loop
}

define void @test2(i32* noalias %loc, i32* noalias %loc2) {
; CHECK-LABEL: @test2
; CHECK: @bar
; CHECK-LABEL: loop:
  br label %loop

loop:
  %res = call i32 @bar(i32* %loc2)
  store i32 %res, i32* %loc
  br label %loop
}

; Negative test: %might clobber gep
define void @test3(i32* %loc) {
; CHECK-LABEL: @test3
; CHECK-LABEL: loop:
; CHECK: @bar
  br label %loop

loop:
  %res = call i32 @bar(i32* %loc)
  %gep = getelementptr i32, i32 *%loc, i64 1000000
  store i32 %res, i32* %gep
  br label %loop
}


; Negative test: %loc might alias %loc2
define void @test4(i32* %loc, i32* %loc2) {
; CHECK-LABEL: @test4
; CHECK-LABEL: loop:
; CHECK: @bar
  br label %loop

loop:
  %res = call i32 @bar(i32* %loc2)
  store i32 %res, i32* %loc
  br label %loop
}

declare i32 @foo_new(i32*) readonly

define void @test5(i32* %loc2, i32* noalias %loc) {
; CHECK-LABEL: @test5
; CHECK: @bar
; CHECK-LABEL: loop:
  br label %loop

loop:
  %res1 = call i32 @bar(i32* %loc2)
  %res = call i32 @foo_new(i32* %loc2)
  store volatile i32 %res1, i32* %loc
  br label %loop
}


; memcpy doesn't write to it's source argument, so loads to that location
; can still be hoisted
define void @test6(i32* noalias %loc, i32* noalias %loc2) {
; CHECK-LABEL: @test6
; CHECK: %val = load i32, i32* %loc2
; CHECK-LABEL: loop:
; CHECK: @llvm.memcpy
  br label %loop

loop:
  %val = load i32, i32* %loc2
  store i32 %val, i32* %loc
  %dest = bitcast i32* %loc to i8*
  %src = bitcast i32* %loc2 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %dest, i8* %src, i64 8, i1 false)
  br label %loop
}

define void @test7(i32* noalias %loc, i32* noalias %loc2) {
; CHECK-LABEL: @test7
; CHECK: %val = load i32, i32* %loc2
; CHECK-LABEL: loop:
; CHECK: @custom_memcpy
  br label %loop

loop:
  %val = load i32, i32* %loc2
  store i32 %val, i32* %loc
  %dest = bitcast i32* %loc to i8*
  %src = bitcast i32* %loc2 to i8*
  call void @custom_memcpy(i8* %dest, i8* %src)
  br label %loop
}

declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture writeonly, i8* nocapture readonly, i64, i1)
declare void @custom_memcpy(i8* nocapture writeonly, i8* nocapture readonly) argmemonly nounwind
