; RUN: opt -aa-pipeline=basic-aa -passes='require<aa>,require<target-ir>,require<scalar-evolution>,require<opt-remark-emit>,loop-mssa(licm)' < %s -S | FileCheck %s

declare i32 @foo() readonly argmemonly nounwind
declare i32 @foo2() readonly nounwind
declare i32 @bar(ptr %loc2) readonly argmemonly nounwind

define void @test(ptr %loc) {
; CHECK-LABEL: @test
; CHECK: @foo
; CHECK-LABEL: loop:
  br label %loop

loop:
  %res = call i32 @foo()
  store i32 %res, ptr %loc
  br label %loop
}

; Negative test: show argmemonly is required
define void @test_neg(ptr %loc) {
; CHECK-LABEL: @test_neg
; CHECK-LABEL: loop:
; CHECK: @foo
  br label %loop

loop:
  %res = call i32 @foo2()
  store i32 %res, ptr %loc
  br label %loop
}

define void @test2(ptr noalias %loc, ptr noalias %loc2) {
; CHECK-LABEL: @test2
; CHECK: @bar
; CHECK-LABEL: loop:
  br label %loop

loop:
  %res = call i32 @bar(ptr %loc2)
  store i32 %res, ptr %loc
  br label %loop
}

; Negative test: %might clobber gep
define void @test3(ptr %loc) {
; CHECK-LABEL: @test3
; CHECK-LABEL: loop:
; CHECK: @bar
  br label %loop

loop:
  %res = call i32 @bar(ptr %loc)
  %gep = getelementptr i32, ptr %loc, i64 1000000
  store i32 %res, ptr %gep
  br label %loop
}


; Negative test: %loc might alias %loc2
define void @test4(ptr %loc, ptr %loc2) {
; CHECK-LABEL: @test4
; CHECK-LABEL: loop:
; CHECK: @bar
  br label %loop

loop:
  %res = call i32 @bar(ptr %loc2)
  store i32 %res, ptr %loc
  br label %loop
}

declare i32 @foo_new(ptr) readonly

define void @test5(ptr %loc2, ptr noalias %loc) {
; CHECK-LABEL: @test5
; CHECK: @bar
; CHECK-LABEL: loop:
  br label %loop

loop:
  %res1 = call i32 @bar(ptr %loc2)
  %res = call i32 @foo_new(ptr %loc2)
  store volatile i32 %res1, ptr %loc
  br label %loop
}


; memcpy doesn't write to it's source argument, so loads to that location
; can still be hoisted
define void @test6(ptr noalias %loc, ptr noalias %loc2) {
; CHECK-LABEL: @test6
; CHECK: %val = load i32, ptr %loc2
; CHECK-LABEL: loop:
; CHECK: @llvm.memcpy
  br label %loop

loop:
  %val = load i32, ptr %loc2
  store i32 %val, ptr %loc
  call void @llvm.memcpy.p0.p0.i64(ptr %loc, ptr %loc2, i64 8, i1 false)
  br label %loop
}

define void @test7(ptr noalias %loc, ptr noalias %loc2) {
; CHECK-LABEL: @test7
; CHECK: %val = load i32, ptr %loc2
; CHECK-LABEL: loop:
; CHECK: @custom_memcpy
  br label %loop

loop:
  %val = load i32, ptr %loc2
  store i32 %val, ptr %loc
  call void @custom_memcpy(ptr %loc, ptr %loc2)
  br label %loop
}

declare void @llvm.memcpy.p0.p0.i64(ptr nocapture writeonly, ptr nocapture readonly, i64, i1)
declare void @custom_memcpy(ptr nocapture writeonly, ptr nocapture readonly) argmemonly nounwind
