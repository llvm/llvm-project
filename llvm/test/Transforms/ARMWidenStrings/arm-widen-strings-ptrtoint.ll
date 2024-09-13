; RUN: opt < %s -mtriple=arm-arm-none-eabi -passes=globalopt -S | FileCheck %s
target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-a:0:32-n32-S64"

; CHECK: [48 x i8]
@f.string1 = private unnamed_addr constant [45 x i8] c"The quick brown dog jumps over the lazy fox.\00", align 1

; Function Attrs: nounwind
define hidden i32 @f() {
entry:
  %string1 = alloca [45 x i8], align 1
  %pos = alloca i32, align 4
  %token = alloca ptr, align 4
  call void @llvm.lifetime.start.p0i8(i64 45, ptr %string1)
  call void @llvm.memcpy.p0i8.p0i8.i32(ptr align 1 %string1, ptr align 1 @f.string1, i32 45, i1 false)
  call void @llvm.lifetime.start.p0i8(i64 4, ptr %pos)
  call void @llvm.lifetime.start.p0i8(i64 4, ptr %token)
  %call = call ptr @strchr(ptr %string1, i32 101)
  store ptr %call, ptr %token, align 4
  %0 = load ptr, ptr %token, align 4
  %sub.ptr.lhs.cast = ptrtoint ptr %0 to i32
  %sub.ptr.rhs.cast = ptrtoint ptr %string1 to i32
  %sub.ptr.sub = sub i32 %sub.ptr.lhs.cast, %sub.ptr.rhs.cast
  %add = add nsw i32 %sub.ptr.sub, 1
  store i32 %add, ptr %pos, align 4
  %1 = load i32, ptr %pos, align 4
  call void @llvm.lifetime.end.p0i8(i64 4, ptr %token)
  call void @llvm.lifetime.end.p0i8(i64 4, ptr %pos)
  call void @llvm.lifetime.end.p0i8(i64 45, ptr %string1)
  ret i32 %1
}

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.start.p0i8(i64, ptr nocapture)

; Function Attrs: argmemonly nounwind
declare void @llvm.memcpy.p0i8.p0i8.i32(ptr nocapture writeonly, ptr nocapture readonly, i32, i1)

; Function Attrs: nounwind
declare ptr @strchr(ptr, i32)

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.end.p0i8(i64, ptr nocapture)
