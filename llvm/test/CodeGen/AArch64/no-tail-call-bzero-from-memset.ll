; RUN: llc -o - %s | FileCheck %s
; RUN: llc -global-isel -o - %s | FileCheck %s
target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"
target triple = "arm64-apple-macosx15.0.0"

define ptr @test()  {
; CHECK-LABEL: test:
; CHECK: bl _bzero
  %1 = tail call ptr @fn(i32 noundef 1) #3
  tail call void @llvm.memset.p0.i64(ptr noundef nonnull align 1 dereferenceable(1000) %1, i8 noundef 0, i64 noundef 1000, i1 noundef false) #3
  ret ptr %1
}

declare ptr @fn(i32 noundef)

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: write)
declare void @llvm.memset.p0.i64(ptr nocapture writeonly, i8, i64, i1 immarg) #2

attributes #2 = { nocallback nofree nounwind willreturn memory(argmem: write) }
attributes #3 = { nounwind optsize }
