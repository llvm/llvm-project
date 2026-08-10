; RUN: opt -S -passes=mergefunc < %s | FileCheck %s
; RUN: opt -S -passes=mergefunc -mergefunc-use-aliases < %s | FileCheck %s -check-prefix=ALIAS

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; ALIAS: @_Z9simple_vaPKcz = unnamed_addr alias void (ptr, ...), ptr @_Z10simple_va2PKcz
; ALIAS-NOT: @_Z9simple_vaPKcz

%struct.__va_list_tag = type { i32, i32, ptr, ptr }

; CHECK-LABEL: define {{.*}}@_Z9simple_vaPKcz
; CHECK: call void @llvm.va_start
; CHECK: call void @llvm.va_end
define dso_local void @_Z9simple_vaPKcz(ptr nocapture readnone, ...) unnamed_addr {
  %2 = alloca [1 x %struct.__va_list_tag], align 16
  call void @llvm.va_start(ptr nonnull %2)
  %3 = load i32, ptr %2, align 16
  %4 = icmp ult i32 %3, 41
  br i1 %4, label %5, label %11

; <label>:7:                                      ; preds = %1
  %6 = getelementptr inbounds [1 x %struct.__va_list_tag], ptr %2, i64 0, i64 0, i32 3
  %7 = load ptr, ptr %6, align 16
  %8 = sext i32 %3 to i64
  %9 = getelementptr i8, ptr %7, i64 %8
  %10 = add i32 %3, 8
  store i32 %10, ptr %2, align 16
  br label %15

; <label>:13:                                     ; preds = %1
  %12 = getelementptr inbounds [1 x %struct.__va_list_tag], ptr %2, i64 0, i64 0, i32 2
  %13 = load ptr, ptr %12, align 8
  %14 = getelementptr i8, ptr %13, i64 8
  store ptr %14, ptr %12, align 8
  br label %15

; <label>:17:                                     ; preds = %11, %5
  %16 = phi ptr [ %9, %5 ], [ %13, %11 ]
  %17 = load i32, ptr %16, align 4
  call void @_Z6escapei(i32 %17)
  call void @llvm.va_end(ptr nonnull %2)
  ret void
}

; Function Attrs: nounwind
declare void @llvm.va_start(ptr)

; Function Attrs: minsize optsize
declare dso_local void @_Z6escapei(i32) local_unnamed_addr

; Function Attrs: nounwind
declare void @llvm.va_end(ptr)

; CHECK-LABEL: define {{.*}}@_Z10simple_va2PKcz
; CHECK: call void @llvm.va_start
; CHECK: call void @llvm.va_end
define dso_local void @_Z10simple_va2PKcz(ptr nocapture readnone, ...) unnamed_addr {
  %2 = alloca [1 x %struct.__va_list_tag], align 16
  call void @llvm.va_start(ptr nonnull %2)
  %3 = load i32, ptr %2, align 16
  %4 = icmp ult i32 %3, 41
  br i1 %4, label %5, label %11

; <label>:7:                                      ; preds = %1
  %6 = getelementptr inbounds [1 x %struct.__va_list_tag], ptr %2, i64 0, i64 0, i32 3
  %7 = load ptr, ptr %6, align 16
  %8 = sext i32 %3 to i64
  %9 = getelementptr i8, ptr %7, i64 %8
  %10 = add i32 %3, 8
  store i32 %10, ptr %2, align 16
  br label %15

; <label>:13:                                     ; preds = %1
  %12 = getelementptr inbounds [1 x %struct.__va_list_tag], ptr %2, i64 0, i64 0, i32 2
  %13 = load ptr, ptr %12, align 8
  %14 = getelementptr i8, ptr %13, i64 8
  store ptr %14, ptr %12, align 8
  br label %15

; <label>:17:                                     ; preds = %11, %5
  %16 = phi ptr [ %9, %5 ], [ %13, %11 ]
  %17 = load i32, ptr %16, align 4
  call void @_Z6escapei(i32 %17)
  call void @llvm.va_end(ptr nonnull %2)
  ret void
}
