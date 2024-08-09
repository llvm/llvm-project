; RUN: opt < %s -passes=asan -asan-dormant -S | FileCheck --check-prefixes=CHECK %s

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: mustprogress noinline nounwind optnone uwtable
define dso_local i32 @test(ptr %a) sanitize_address {
entry:
  %a.addr = alloca ptr, align 8
  store ptr %a, ptr %a.addr, align 8
  %0 = load ptr, ptr %a.addr, align 8
  store i32 5, ptr %0, align 4
  %1 = load ptr, ptr %a.addr, align 8
  %2 = load i32, ptr %1, align 4
  ret i32 %2


; CHECK: %a.addr = alloca ptr, align 8
; CHECK: store ptr %a, ptr %a.addr, align 8
; CHECK:   %0 = load ptr, ptr %a.addr, align 8
; CHECK:   %1 = load i1, ptr @__asan_is_dormant, align 1
; CHECK:   %2 = xor i1 %1, true
; CHECK:   br i1 %2, label %3, label %17

; CHECK:   %4 = ptrtoint ptr %0 to i64
; CHECK:   %5 = lshr i64 %4, 3
; CHECK:   %6 = add i64 %5, 2147450880
; CHECK:   %7 = inttoptr i64 %6 to ptr
; CHECK:   %8 = load i8, ptr %7, align 1
; CHECK:   %9 = icmp ne i8 %8, 0
; CHECK:   br i1 %9, label %10, label %16, !prof !6

; CHECK:   %11 = and i64 %4, 7
; CHECK:   %12 = add i64 %11, 3
; CHECK:   %13 = trunc i64 %12 to i8
; CHECK:   %14 = icmp sge i8 %13, %8
; CHECK:   br i1 %14, label %15, label %16

; CHECK:   call void @__asan_report_store4(i64 %4) #3
; CHECK:   unreachable

; CHECK:   br label %17

; CHECK:   store i32 5, ptr %0, align 4
; CHECK:   %18 = load ptr, ptr %a.addr, align 8
; CHECK:   %19 = load i1, ptr @__asan_is_dormant, align 1
; CHECK:   %20 = xor i1 %19, true
; CHECK:   br i1 %20, label %21, label %35

; CHECK:   %22 = ptrtoint ptr %18 to i64
; CHECK:   %23 = lshr i64 %22, 3
; CHECK:   %24 = add i64 %23, 2147450880
; CHECK:   %25 = inttoptr i64 %24 to ptr
; CHECK:   %26 = load i8, ptr %25, align 1
; CHECK:   %27 = icmp ne i8 %26, 0
; CHECK:   br i1 %27, label %28, label %34, !prof !6

; CHECK:   %29 = and i64 %22, 7
; CHECK:   %30 = add i64 %29, 3
; CHECK:   %31 = trunc i64 %30 to i8
; CHECK:   %32 = icmp sge i8 %31, %26
; CHECK:   br i1 %32, label %33, label %34

; CHECK:   call void @__asan_report_load4(i64 %22) #3
; CHECK:   unreachable

; CHECK:   br label %35

; CHECK:   %36 = load i32, ptr %18, align 4
; CHECK:   ret i32 %36

}

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 2}
!5 = !{!"clang version 20.0.0git (https://github.com/gbMattN/llvm-project.git 3d0736cd0a60f7f0f78a14982091e5687e2be7da)"}
