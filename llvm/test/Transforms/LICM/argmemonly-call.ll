; RUN: opt -S -basicaa -licm %s | FileCheck %s
; RUN: opt -aa-pipeline=basic-aa -passes='require<aa>,require<targetir>,require<scalar-evolution>,require<opt-remark-emit>,loop(licm)' < %s -S | FileCheck %s
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
