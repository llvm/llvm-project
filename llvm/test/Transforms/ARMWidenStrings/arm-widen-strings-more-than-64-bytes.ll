; RUN: opt < %s -mtriple=arm-arm-none-eabi -passes=globalopt -S | FileCheck %s
target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-a:0:32-n32-S64"
target triple = "thumbv6m-arm-none-eabi"

; CHECK: [65 x i8]
; CHECK-NOT: [68 x i8]
@.str = private unnamed_addr constant [65 x i8] c"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaazzz\00", align 1

; Function Attrs: nounwind
define hidden void @foo() local_unnamed_addr #0 {
entry:
  %something = alloca [65 x i8], align 1
  call void @llvm.lifetime.start(i64 65, ptr nonnull %something) #3
  call void @llvm.memcpy.p0i8.p0i8.i32(ptr align 1 nonnull %something, ptr align 1 @.str, i32 65, i1 false)
  %call2 = call i32 @bar(ptr nonnull %something) #3
  call void @llvm.lifetime.end(i64 65, ptr nonnull %something) #3
  ret void
}

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.start(i64, ptr nocapture) #1

declare i32 @bar(...) local_unnamed_addr #2

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.end(i64, ptr nocapture) #1

; Function Attrs: argmemonly nounwind
declare void @llvm.memcpy.p0i8.p0i8.i32(ptr nocapture writeonly, ptr nocapture readonly, i32, i1) #1
