; RUN: opt -S -mergefunc < %s | FileCheck %s
; RUN: opt -S -mergefunc -mergefunc-use-aliases < %s | FileCheck %s -check-prefix=ALIAS

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; ALIAS: @_Z9simple_vaPKcz = unnamed_addr alias void (i8*, ...), void (i8*, ...)* @_Z10simple_va2PKcz
; ALIAS-NOT: @_Z9simple_vaPKcz

%struct.__va_list_tag = type { i32, i32, i8*, i8* }

; CHECK-LABEL: define {{.*}}@_Z9simple_vaPKcz
; CHECK: call void @llvm.va_start
; CHECK: call void @llvm.va_end
define dso_local void @_Z9simple_vaPKcz(i8* nocapture readnone, ...) unnamed_addr {
  %2 = alloca [1 x %struct.__va_list_tag], align 16
  %3 = bitcast [1 x %struct.__va_list_tag]* %2 to i8*
  call void @llvm.va_start(i8* nonnull %3)
  %4 = getelementptr inbounds [1 x %struct.__va_list_tag], [1 x %struct.__va_list_tag]* %2, i64 0, i64 0, i32 0
  %5 = load i32, i32* %4, align 16
  %6 = icmp ult i32 %5, 41
  br i1 %6, label %7, label %13

; <label>:7:                                      ; preds = %1
  %8 = getelementptr inbounds [1 x %struct.__va_list_tag], [1 x %struct.__va_list_tag]* %2, i64 0, i64 0, i32 3
  %9 = load i8*, i8** %8, align 16
  %10 = sext i32 %5 to i64
  %11 = getelementptr i8, i8* %9, i64 %10
  %12 = add i32 %5, 8
  store i32 %12, i32* %4, align 16
  br label %17

; <label>:13:                                     ; preds = %1
  %14 = getelementptr inbounds [1 x %struct.__va_list_tag], [1 x %struct.__va_list_tag]* %2, i64 0, i64 0, i32 2
  %15 = load i8*, i8** %14, align 8
  %16 = getelementptr i8, i8* %15, i64 8
  store i8* %16, i8** %14, align 8
  br label %17

; <label>:17:                                     ; preds = %13, %7
  %18 = phi i8* [ %11, %7 ], [ %15, %13 ]
  %19 = bitcast i8* %18 to i32*
  %20 = load i32, i32* %19, align 4
  call void @_Z6escapei(i32 %20)
  call void @llvm.va_end(i8* nonnull %3)
  ret void
}

; Function Attrs: nounwind
declare void @llvm.va_start(i8*)

; Function Attrs: minsize optsize
declare dso_local void @_Z6escapei(i32) local_unnamed_addr

; Function Attrs: nounwind
declare void @llvm.va_end(i8*)

; CHECK-LABEL: define {{.*}}@_Z10simple_va2PKcz
; CHECK: call void @llvm.va_start
; CHECK: call void @llvm.va_end
define dso_local void @_Z10simple_va2PKcz(i8* nocapture readnone, ...) unnamed_addr {
  %2 = alloca [1 x %struct.__va_list_tag], align 16
  %3 = bitcast [1 x %struct.__va_list_tag]* %2 to i8*
  call void @llvm.va_start(i8* nonnull %3)
  %4 = getelementptr inbounds [1 x %struct.__va_list_tag], [1 x %struct.__va_list_tag]* %2, i64 0, i64 0, i32 0
  %5 = load i32, i32* %4, align 16
  %6 = icmp ult i32 %5, 41
  br i1 %6, label %7, label %13

; <label>:7:                                      ; preds = %1
  %8 = getelementptr inbounds [1 x %struct.__va_list_tag], [1 x %struct.__va_list_tag]* %2, i64 0, i64 0, i32 3
  %9 = load i8*, i8** %8, align 16
  %10 = sext i32 %5 to i64
  %11 = getelementptr i8, i8* %9, i64 %10
  %12 = add i32 %5, 8
  store i32 %12, i32* %4, align 16
  br label %17

; <label>:13:                                     ; preds = %1
  %14 = getelementptr inbounds [1 x %struct.__va_list_tag], [1 x %struct.__va_list_tag]* %2, i64 0, i64 0, i32 2
  %15 = load i8*, i8** %14, align 8
  %16 = getelementptr i8, i8* %15, i64 8
  store i8* %16, i8** %14, align 8
  br label %17

; <label>:17:                                     ; preds = %13, %7
  %18 = phi i8* [ %11, %7 ], [ %15, %13 ]
  %19 = bitcast i8* %18 to i32*
  %20 = load i32, i32* %19, align 4
  call void @_Z6escapei(i32 %20)
  call void @llvm.va_end(i8* nonnull %3)
  ret void
}
