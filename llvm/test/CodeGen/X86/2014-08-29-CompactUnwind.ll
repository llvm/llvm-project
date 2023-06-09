; RUN: llc < %s -mtriple x86_64-apple-darwin11 -mcpu corei7 -emit-compact-unwind-non-canonical=true -filetype=obj -o - | llvm-objdump -d --unwind-info -s - | FileCheck %s
; Regression test for http://llvm.org/bugs/show_bug.cgi?id=20800.

; ModuleID = 'asan_report.ii'
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.9.0"

@.str = private unnamed_addr constant [3 x i8] c"=>\00", align 1
@.str1 = private unnamed_addr constant [3 x i8] c"  \00", align 1
@.str2 = private unnamed_addr constant [6 x i8] c"%s%p:\00", align 1

; CHECK: <___asan_report_error>:

; subq instruction starts at 0x0a, so the second byte of the compact encoding
; (UNWIND_X86_64_FRAMELESS_STACK_SIZE in mach-o/compact_unwind_encoding.h)
; must be 0x0d.
; CHECK: {{a:.*subq.*%rsp}}

; CHECK: Contents of __compact_unwind section
; CHECK: ___asan_report_error

; Because of incorrect push instruction size in X86AsmBackend.cpp the stack
; size was also calculated incorrectly.
; CHECK-NOT: {{compact encoding:.*0x0309f800}}
; CHECK: {{compact encoding:.*0x030df800}}

define void @__asan_report_error(i64 %step) #0 {
  %str.i = alloca i64, align 8
  %stack = alloca [256 x i64], align 8
  br label %print_shadow_bytes.exit.i

print_shadow_bytes.exit.i: ; preds = %print_shadow_bytes.exit.i, %0
  %iv.i = phi i64 [ -5, %0 ], [ %iv.next.i, %print_shadow_bytes.exit.i ]
  %reg15 = icmp eq i64 %iv.i, 0
  %.str..str1.i = select i1 %reg15, [3 x i8]* @.str, [3 x i8]* @.str1
  %reg16 = getelementptr inbounds [3 x i8], [3 x i8]* %.str..str1.i, i64 0, i64 0
  %reg17 = shl i64 %iv.i, 1
  %reg19 = inttoptr i64 %reg17 to i8*
  call void (i64*, i8*, ...) @append(i64* %str.i, i8* getelementptr inbounds ([6 x i8], [6 x i8]* @.str2, i64 0, i64 0), i8* %reg16, i8* %reg19)
  %iv.next.i = add nsw i64 %iv.i, %step
  br label %print_shadow_bytes.exit.i
}

declare void @append(i64*, i8*, ...)

attributes #0 = { "frame-pointer"="none" }
