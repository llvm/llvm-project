; RUN: opt %s -o %t.bc
; RUN: llvm-lto2 run %t.bc -r %t.bc,bar,px -r %t.bc,foo,px -r %t.bc,test,px \
; RUN: -r %t.bc,fptr1,px -r %t.bc,fptr2,px -save-temps -o %t.o
; RUN: llvm-dis %t.*.opt.bc

; REM: Find the `llvm.global.annotations` symbol in `%t.*.opt.ll` and
; REM: verify that no function annotation references CFI jump table entry.

; RUN: grep llvm.global.annotations %t.*.opt.ll > %t.annotations
; RUN: grep bar %t.annotations
; RUN: grep foo %t.annotations
; RUN: not grep cfi %t.annotations

; ModuleID = 'cfi-annotation'
source_filename = "ld-temp.o"
target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64-none-linux-gnu"

@llvm.global.annotations = appending global [2 x { ptr, ptr, ptr, i32, ptr }] [{ ptr, ptr, ptr, i32, ptr } { ptr @bar, ptr @.str, ptr @.str.1, i32 4, ptr null }, { ptr, ptr, ptr, i32, ptr } { ptr @foo, ptr @.str.2, ptr @.str.1, i32 3, ptr null }], section "llvm.metadata"
@fptr1 = global ptr null, align 8
@fptr2 = global ptr null, align 8
@.str = private unnamed_addr constant [9 x i8] c"test_bar\00", section "llvm.metadata"
@.str.1 = private unnamed_addr constant [17 x i8] c"cfi-annotation.c\00", section "llvm.metadata"
@.str.2 = private unnamed_addr constant [9 x i8] c"test_foo\00", section "llvm.metadata"

; Function Attrs: noinline nounwind optnone uwtable
define i32 @bar(i32 noundef %0) #0 !type !8 !type !9 {
  %2 = alloca i32, align 4
  store i32 %0, ptr %2, align 4
  %3 = load i32, ptr %2, align 4
  %4 = add nsw i32 %3, -1
  store i32 %4, ptr %2, align 4
  %5 = call i32 @foo(i32 noundef %4)
  %6 = load i32, ptr %2, align 4
  %7 = add nsw i32 %6, 1
  store i32 %7, ptr %2, align 4
  %8 = call i32 @foo(i32 noundef %7)
  %9 = add nsw i32 %5, %8
  ret i32 %9
}

declare !type !8 !type !9 i32 @foo(i32 noundef) #1

; Function Attrs: noinline nounwind optnone uwtable
define i32 @test(i32 noundef %0) #0 !type !8 !type !9 {
  %2 = alloca i32, align 4
  store i32 %0, ptr %2, align 4
  %3 = load i32, ptr %2, align 4
  %4 = icmp sgt i32 %3, 0
  br i1 %4, label %5, label %6

5:                                                ; preds = %1
  store ptr @bar, ptr @fptr1, align 8
  store ptr @foo, ptr @fptr2, align 8
  br label %7

6:                                                ; preds = %1
  store ptr @bar, ptr @fptr1, align 8
  store ptr @foo, ptr @fptr2, align 8
  br label %7

7:                                                ; preds = %6, %5
  %8 = load ptr, ptr @fptr1, align 8
  %9 = call i1 @llvm.type.test(ptr %8, metadata !"_ZTSFiiE"), !nosanitize !10
  br i1 %9, label %11, label %10, !nosanitize !10

10:                                               ; preds = %7
  call void @llvm.ubsantrap(i8 2) #4, !nosanitize !10
  unreachable, !nosanitize !10

11:                                               ; preds = %7
  %12 = load ptr, ptr @fptr2, align 8
  %13 = call i1 @llvm.type.test(ptr %12, metadata !"_ZTSFiiE"), !nosanitize !10
  br i1 %13, label %15, label %14, !nosanitize !10

14:                                               ; preds = %11
  call void @llvm.ubsantrap(i8 2) #4, !nosanitize !10
  unreachable, !nosanitize !10

15:                                               ; preds = %11
  %16 = load i32, ptr %2, align 4
  %17 = call i32 %12(i32 noundef %16)
  %18 = call i32 %8(i32 noundef %17)
  ret i32 %18
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i1 @llvm.type.test(ptr, metadata) #2

; Function Attrs: cold noreturn nounwind
declare void @llvm.ubsantrap(i8 immarg) #3

attributes #0 = { noinline nounwind optnone uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+v8a,-fmv" }
attributes #1 = { "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+v8a,-fmv" }
attributes #2 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #3 = { cold noreturn nounwind }
attributes #4 = { noreturn nounwind }

!llvm.ident = !{!0}
!llvm.module.flags = !{!1, !2, !3, !4, !5, !6, !7}

!0 = !{!"clang version 19.0.0git (https://github.com/yozhu/llvm-project.git a6fef79167066afdf715c6f1bb7834a9d04d575e)"}
!1 = !{i32 1, !"wchar_size", i32 4}
!2 = !{i32 4, !"CFI Canonical Jump Tables", i32 0}
!3 = !{i32 8, !"PIC Level", i32 2}
!4 = !{i32 7, !"uwtable", i32 2}
!5 = !{i32 7, !"frame-pointer", i32 1}
!6 = !{i32 1, !"ThinLTO", i32 0}
!7 = !{i32 1, !"EnableSplitLTOUnit", i32 1}
!8 = !{i64 0, !"_ZTSFiiE"}
!9 = !{i64 0, !"_ZTSFiiE.generalized"}
!10 = !{}
