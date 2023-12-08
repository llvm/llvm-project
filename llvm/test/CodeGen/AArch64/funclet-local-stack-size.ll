; RUN: llc -o - %s -mtriple=aarch64-windows | FileCheck %s
; Check that the local stack size is computed correctly for a funclet contained
; within a varargs function.  The varargs component shouldn't be included in the
; local stack size computation.
target datalayout = "e-m:w-p:64:64-i32:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64-unknown-windows-msvc19.11.0"

%rtti.TypeDescriptor2 = type { ptr, ptr, [3 x i8] }

$"??_R0H@8" = comdat any

@"??_7type_info@@6B@" = external constant ptr
@"??_R0H@8" = linkonce_odr global %rtti.TypeDescriptor2 { ptr @"??_7type_info@@6B@", ptr null, [3 x i8] c".H\00" }, comdat

; CHECK-LABEL: ?catch$2@?0??func@@YAHHHZZ@4HA
; CHECK: stp x29, x30, [sp, #-16]!
; CHECK: ldp x29, x30, [sp], #16
; Function Attrs: uwtable
define dso_local i32 @"?func@@YAHHHZZ"(i32 %a, i32, ...) local_unnamed_addr #0 personality ptr @__CxxFrameHandler3 {
entry:
  %arr = alloca [10 x i32], align 4
  %a2 = alloca i32, align 4
  %call = call i32 @"?init@@YAHPEAH@Z"(ptr nonnull %arr)
  %call1 = invoke i32 @"?func2@@YAHXZ"()
          to label %cleanup unwind label %catch.dispatch

catch.dispatch:                                   ; preds = %entry
  %1 = catchswitch within none [label %catch] unwind to caller

catch:                                            ; preds = %catch.dispatch
  %2 = catchpad within %1 [ptr @"??_R0H@8", i32 0, ptr %a2]
  %3 = load i32, ptr %a2, align 4
  %add = add nsw i32 %3, 1
  catchret from %2 to label %cleanup

cleanup:                                          ; preds = %entry, %catch
  %retval.0 = phi i32 [ %add, %catch ], [ %call1, %entry ]
  ret i32 %retval.0
}

declare dso_local i32 @"?init@@YAHPEAH@Z"(ptr)

declare dso_local i32 @"?func2@@YAHXZ"()

declare dso_local i32 @__CxxFrameHandler3(...)

attributes #0 = { uwtable }

!llvm.module.flags = !{!0}

!0 = !{i32 1, !"wchar_size", i32 2}
