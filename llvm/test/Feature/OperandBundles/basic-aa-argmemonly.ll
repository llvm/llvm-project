; RUN: opt -S -basicaa -gvn < %s | FileCheck %s

declare void @argmemonly_function(i32 *) argmemonly

define i32 @test0(i32* %P, i32* noalias %P2) {
; CHECK-LABEL: @test0(
  %v1 = load i32, i32* %P
; CHECK: %v1 = load i32, i32* %P
  call void @argmemonly_function(i32* %P2) [ "tag"() ]
; CHECK: call void @argmemonly_function(
  %v2 = load i32, i32* %P
; CHECK: %v2 = load i32, i32* %P
  %diff = sub i32 %v1, %v2
; CHECK: %diff = sub i32 %v1, %v2
  ret i32 %diff
; CHECK: ret i32 %diff
}

define i32 @test1(i32* %P, i32* noalias %P2) {
; CHECK-LABEL: @test1(
  %v1 = load i32, i32* %P
  call void @argmemonly_function(i32* %P2) argmemonly [ "tag"() ]
; CHECK: call void @argmemonly_function(
  %v2 = load i32, i32* %P
  %diff = sub i32 %v1, %v2
  ret i32 %diff
; CHECK: ret i32 0
}
