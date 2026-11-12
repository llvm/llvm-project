; RUN: llc -fast-isel=1 -mcpu=ppc64 -mtriple=powerpc64 < %s | FileCheck %s
; Check for non immediate compare insn.

; ModuleID = 'test.c'
source_filename = "test.c"
target datalayout = "E-m:e-i64:64-n32:64"
target triple = "ppc64"

@.str = private unnamed_addr constant [9 x i8] c"correct\0A\00", align 1
@.str.1 = private unnamed_addr constant [11 x i8] c"incorrect\0A\00", align 1

; Function Attrs: noinline nounwind optnone uwtable
define dso_local signext i32 @myTest() #0 {
  %1 = alloca i64, align 8
  %2 = alloca i64, align 8
  store i64 4660, ptr %1, align 8
  store i64 140737488355328, ptr %2, align 8
  %3 = load i64, ptr %1, align 8
  %4 = icmp ult i64 %3, 140737488355328
  br i1 %4, label %5, label %7

5:                                                ; preds = %0
  %6 = call signext i32 (ptr, ...) @printf(ptr noundef @.str)
  br label %9

7:                                                ; preds = %0
  %8 = call signext i32 (ptr, ...) @printf(ptr noundef @.str.1)
  br label %9

9:                                                ; preds = %7, %5
  ret i32 0
}

declare signext i32 @printf(ptr noundef, ...) #1

; CHECK-LABEL: myTest:
; CHECK:       # %bb.0:
; CHECK:       mflr 0
; CHECK:       li 3, 1
; CHECK:       sldi 3, 3, 47
; CHECK:       ld 4, 120(1)
; CHECK:       cmpld   4, 3
