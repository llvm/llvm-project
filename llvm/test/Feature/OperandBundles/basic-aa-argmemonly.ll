; RUN: opt -S -aa-pipeline=basic-aa -passes=gvn < %s | FileCheck %s

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

define i32 @test2(i32* %P, i32* noalias %P2) {
; Note: in this test we //can// GVN %v1 and %v2 into one value in theory.  Calls
; with deopt operand bundles are not argmemonly because they *read* the entire
; heap, but they don't write to any location in the heap if the callee does not
; deoptimize the caller.  This fact, combined with the fact that
; @argmemonly_function is, well, an argmemonly function, can be used to conclude
; that %P is not written to at the callsite.  However LLVM currently cannot
; describe the "does not write to non-args, and reads the entire heap" effect on
; a callsite.

; CHECK-LABEL: @test2(
  %v1 = load i32, i32* %P
; CHECK: %v1 = load i32, i32* %P
  call void @argmemonly_function(i32* %P2) [ "deopt"() ]
; CHECK: call void @argmemonly_function(
  %v2 = load i32, i32* %P
; CHECK: %v2 = load i32, i32* %P
  %diff = sub i32 %v1, %v2
; CHECK: %diff = sub i32 %v1, %v2
  ret i32 %diff
; CHECK: ret i32 %diff
}
