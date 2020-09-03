; RUN: llc -o - %s -mtriple=aarch64-windows | FileCheck %s
; Check that the stack bump around a funclet is computed correctly in both the
; prologue and epilogue in the case we have a MaxCallFrameSize > 0 and are doing alloca
target datalayout = "e-m:w-p:64:64-i32:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64-pc-windows-msvc19.25.28611"

; // requires passing arguments on the stack
; void test2(void*, int, int, int, int, int, int, int, int);
;
; // function with the funclet being checked
; void test1(size_t bytes)
; {
;   // alloca forces a separate callee save bump and stack bump
;   void *data = _alloca(bytes);
;   try {
;     test2(data, 0, 1, 2, 3, 4, 5, 6, 7);
;   } catch (...) {
;     // the funclet being checked
;   }
; }

; CHECK-LABEL: ?catch$2@?0??test1@@YAX_K@Z@4HA
; CHECK: sub sp, sp, #16
; CHECK: add sp, sp, #16
; Function Attrs: uwtable
define dso_local void @"?test1@@YAX_K@Z"(i64 %0) #0 personality i8* bitcast (i32 (...)* @__CxxFrameHandler3 to i8*) {
  %2 = alloca i64, align 8
  %3 = alloca i8*, align 8
  store i64 %0, i64* %2, align 8
  %4 = load i64, i64* %2, align 8
  %5 = alloca i8, i64 %4, align 16
  store i8* %5, i8** %3, align 8
  %6 = load i8*, i8** %3, align 8
  invoke void @"?test2@@YAXPEAXHHHHHHHH@Z"(i8* %6, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7)
          to label %13 unwind label %7

7:                                                ; preds = %1
  %8 = catchswitch within none [label %9] unwind to caller

9:                                                ; preds = %7
  %10 = catchpad within %8 [i8* null, i32 64, i8* null]
  catchret from %10 to label %11

11:                                               ; preds = %9
  br label %12

12:                                               ; preds = %11, %13
  ret void

13:                                               ; preds = %1
  br label %12
}

declare dso_local void @"?test2@@YAXPEAXHHHHHHHH@Z"(i8*, i32, i32, i32, i32, i32, i32, i32, i32) #1

declare dso_local i32 @__CxxFrameHandler3(...)

attributes #0 = { uwtable }

!llvm.module.flags = !{!0}

!0 = !{i32 1, !"wchar_size", i32 2}
