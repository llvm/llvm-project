; RUN: llc -o - %s -mtriple=aarch64-windows -verify-machineinstrs | FileCheck %s

; Check we don't have a base pointer.
; CHECK-LABEL: "?a@@YAXXZ":
; CHECK-NOT: x19

; Make sure the prologue and epilogue are sane. Make sure the
; frame index is relative to the FP, since there is no base pointer.
; (Funclets aren't allowed to contain dynamic allocas.)
; CHECK-LABEL: "?catch$2@?0??a@@YAXXZ@4HA":
; CHECK:      stp     x29, x30, [sp, #-16]!
; CHECK-NEXT: .seh_save_fplr_x 16
; CHECK-NEXT: .seh_endprologue
; CHECK-NEXT: ldur    x0, [x29, #-8]
; CHECK-NEXT: mov     x1, x0
; CHECK-NEXT: bl      "?bb@@YAXPEAHH@Z"
; CHECK-NEXT: adrp    x0, .LBB0_1
; CHECK-NEXT: add     x0, x0, .LBB0_1
; CHECK-NEXT: .seh_startepilogue
; CHECK-NEXT: ldp     x29, x30, [sp], #16
; CHECK-NEXT: .seh_save_fplr_x 16
; CHECK-NEXT: .seh_endepilogue
; CHECK-NEXT: ret

target datalayout = "e-m:w-p:64:64-i32:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64-unknown-windows-msvc19.11.0"

define dso_local void @"?a@@YAXXZ"(i64 %p1) personality i8* bitcast (i32 (...)* @__CxxFrameHandler3 to i8*) {
entry:
  %a = alloca i32, i64 %p1, align 16
  %0 = bitcast i32* %a to i8*  
  call void @llvm.memset.p0i8.i64(i8* nonnull align 16 %0, i8 0, i64 400, i1 false)
  store i32 305419896, i32* %a, align 16
  invoke void @"?bb@@YAXPEAHH@Z"(i32* nonnull %a, i32* null)
          to label %try.cont unwind label %catch.dispatch

catch.dispatch:                                   ; preds = %entry
  %1 = catchswitch within none [label %catch] unwind to caller

catch:                                            ; preds = %catch.dispatch
  %2 = catchpad within %1 [i8* null, i32 64, i8* null]
  call void @"?bb@@YAXPEAHH@Z"(i32* nonnull %a, i32* %a) [ "funclet"(token %2) ]
  catchret from %2 to label %try.cont

try.cont:                                         ; preds = %entry, %catch
  call void @"?cc@@YAXXZ"()
  ret void
}

declare void @llvm.memset.p0i8.i64(i8* nocapture writeonly, i8, i64, i1)

declare dso_local void @"?bb@@YAXPEAHH@Z"(i32*, i32*)

declare dso_local i32 @__CxxFrameHandler3(...)

declare dso_local void @"?cc@@YAXXZ"()

