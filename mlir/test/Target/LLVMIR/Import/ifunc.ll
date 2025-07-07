; RUN: mlir-translate --import-llvm %s -split-input-file | FileCheck %s

@__const.main.data = private unnamed_addr constant [10 x i32] [i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10], align 16

; CHECK: llvm.mlir.ifunc @foo : !llvm.func<void (ptr, i64)>, !llvm.ptr @resolve_foo {dso_local}
@foo = dso_local ifunc void (ptr, i64), ptr @resolve_foo

define dso_local void @foo_1(ptr noundef %0, i64 noundef %1) #0 {
  %3 = alloca ptr, align 8
  %4 = alloca i64, align 8
  store ptr %0, ptr %3, align 8
  store i64 %1, ptr %4, align 8
  ret void
}

define dso_local void @foo_2(ptr noundef %0, i64 noundef %1) #0 {
  %3 = alloca ptr, align 8
  %4 = alloca i64, align 8
  store ptr %0, ptr %3, align 8
  store i64 %1, ptr %4, align 8
  ret void
}

define dso_local i32 @main() #0 {
  %1 = alloca [10 x i32], align 16
  call void @llvm.memcpy.p0.p0.i64(ptr align 16 %1, ptr align 16 @__const.main.data, i64 40, i1 false)
  %2 = getelementptr inbounds [10 x i32], ptr %1, i64 0, i64 0
; CHECK: llvm.call @foo
  call void @foo(ptr noundef %2, i64 noundef 10)
  ret i32 0
}

declare void @llvm.memcpy.p0.p0.i64(ptr noalias nocapture writeonly, ptr noalias nocapture readonly, i64, i1 immarg) #1

define internal ptr @resolve_foo() #2 {
  %1 = alloca ptr, align 8
  %2 = call i32 @check()
  %3 = icmp ne i32 %2, 0
  br i1 %3, label %4, label %5

4:                                                ; preds = %0
  store ptr @foo_1, ptr %1, align 8
  br label %6

5:                                                ; preds = %0
  store ptr @foo_2, ptr %1, align 8
  br label %6

6:                                                ; preds = %5, %4
  %7 = load ptr, ptr %1, align 8
  ret ptr %7
}

declare i32 @check() #3

; // -----

@__const.main.data = private unnamed_addr constant [10 x i32] [i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10], align 16

; CHECK: llvm.mlir.ifunc @foo : !llvm.func<void (ptr, i64)>, !llvm.ptr @resolve_foo {dso_local}
@foo = dso_local ifunc void (ptr, i64), ptr @resolve_foo

define dso_local void @foo_1(ptr noundef %0, i64 noundef %1) #0 {
  %3 = alloca ptr, align 8
  %4 = alloca i64, align 8
  store ptr %0, ptr %3, align 8
  store i64 %1, ptr %4, align 8
  ret void
}

define dso_local void @foo_2(ptr noundef %0, i64 noundef %1) #0 {
  %3 = alloca ptr, align 8
  %4 = alloca i64, align 8
  store ptr %0, ptr %3, align 8
  store i64 %1, ptr %4, align 8
  ret void
}

define dso_local i32 @main() #0 {
  %1 = alloca [10 x i32], align 16
  %2 = alloca ptr, align 8
  call void @llvm.memcpy.p0.p0.i64(ptr align 16 %1, ptr align 16 @__const.main.data, i64 40, i1 false)
; CHECK: [[CALLEE:%[0-9]+]] = llvm.mlir.addressof @foo
; CHECK: llvm.store [[CALLEE]], [[STORED:%[0-9]+]]
; CHECK: [[LOADED_CALLEE:%[0-9]+]] = llvm.load [[STORED]]
  store ptr @foo, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  %4 = getelementptr inbounds [10 x i32], ptr %1, i64 0, i64 0
; CHECK: llvm.call [[LOADED_CALLEE]]
  call void %3(ptr noundef %4, i64 noundef 10)
  ret i32 0
}

declare void @llvm.memcpy.p0.p0.i64(ptr noalias nocapture writeonly, ptr noalias nocapture readonly, i64, i1 immarg) #1

define internal ptr @resolve_foo() #2 {
  %1 = alloca ptr, align 8
  %2 = call i32 @check()
  %3 = icmp ne i32 %2, 0
  br i1 %3, label %4, label %5

4:                                                ; preds = %0
  store ptr @foo_1, ptr %1, align 8
  br label %6

5:                                                ; preds = %0
  store ptr @foo_2, ptr %1, align 8
  br label %6

6:                                                ; preds = %5, %4
  %7 = load ptr, ptr %1, align 8
  ret ptr %7
}

declare i32 @check() #3
