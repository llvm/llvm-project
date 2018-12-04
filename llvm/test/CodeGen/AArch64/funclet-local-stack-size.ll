; RUN: llc -o - %s -mtriple=aarch64-windows | FileCheck %s
; Check that the local stack size is computed correctly for a funclet contained
; within a varargs function.  The varargs component shouldn't be included in the
; local stack size computation.
target datalayout = "e-m:w-p:64:64-i32:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64-unknown-windows-msvc19.11.0"

%rtti.TypeDescriptor2 = type { i8**, i8*, [3 x i8] }

$"??_R0H@8" = comdat any

@"??_7type_info@@6B@" = external constant i8*
@"??_R0H@8" = linkonce_odr global %rtti.TypeDescriptor2 { i8** @"??_7type_info@@6B@", i8* null, [3 x i8] c".H\00" }, comdat

; CHECK-LABEL: ?catch$2@?0??func@@YAHHHZZ@4HA
; CHECK: stp x29, x30, [sp, #-16]!
; CHECK: ldp x29, x30, [sp], #16
; Function Attrs: uwtable
define dso_local i32 @"?func@@YAHHHZZ"(i32 %a, i32, ...) local_unnamed_addr #0 personality i8* bitcast (i32 (...)* @__CxxFrameHandler3 to i8*) {
entry:
  %arr = alloca [10 x i32], align 4
  %a2 = alloca i32, align 4
  %1 = bitcast [10 x i32]* %arr to i8*
  %arraydecay = getelementptr inbounds [10 x i32], [10 x i32]* %arr, i64 0, i64 0
  %call = call i32 @"?init@@YAHPEAH@Z"(i32* nonnull %arraydecay)
  %call1 = invoke i32 @"?func2@@YAHXZ"()
          to label %cleanup unwind label %catch.dispatch

catch.dispatch:                                   ; preds = %entry
  %2 = catchswitch within none [label %catch] unwind to caller

catch:                                            ; preds = %catch.dispatch
  %3 = catchpad within %2 [%rtti.TypeDescriptor2* @"??_R0H@8", i32 0, i32* %a2]
  %4 = load i32, i32* %a2, align 4
  %add = add nsw i32 %4, 1
  catchret from %3 to label %cleanup

cleanup:                                          ; preds = %entry, %catch
  %retval.0 = phi i32 [ %add, %catch ], [ %call1, %entry ]
  ret i32 %retval.0
}

declare dso_local i32 @"?init@@YAHPEAH@Z"(i32*)

declare dso_local i32 @"?func2@@YAHXZ"()

declare dso_local i32 @__CxxFrameHandler3(...)

attributes #0 = { uwtable }

!llvm.module.flags = !{!0}

!0 = !{i32 1, !"wchar_size", i32 2}
