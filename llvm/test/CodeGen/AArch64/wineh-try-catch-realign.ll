; RUN: llc -o - %s -mtriple=aarch64-windows -verify-machineinstrs | FileCheck %s

; Make sure we have a base pointer.
; CHECK-LABEL: "?a@@YAXXZ":
; CHECK: and     sp, x9, #0xffffffffffffffc0
; CHECK: mov     x19, sp

; Make sure the funclet prologue/epilogue are correct: specifically,
; it shouldn't access the parent's frame via sp, and the prologue and
; epilogue should be symmetrical.
; CHECK-LABEL: "?catch$2@?0??a@@YAXXZ@4HA":
; CHECK:      stp     x29, x30, [sp, #-32]!
; CHECK-NEXT: .seh_save_fplr_x 32
; CHECK-NEXT: str     x28, [sp, #16]
; CHECK-NEXT: .seh_save_reg x28, 16
; CHECK-NEXT: str     x19, [sp, #24]
; CHECK-NEXT: .seh_save_reg x19, 24
; CHECK-NEXT: .seh_endprologue
; CHECK-NEXT: add     x0, x19, #0
; CHECK-NEXT: mov     w1, wzr
; CHECK-NEXT: bl      "?bb@@YAXPEAHH@Z"
; CHECK-NEXT: adrp    x0, .LBB0_1
; CHECK-NEXT: add     x0, x0, .LBB0_1
; CHECK-NEXT: .seh_startepilogue
; CHECK-NEXT: ldr     x19, [sp, #24]
; CHECK-NEXT: .seh_save_reg x19, 24
; CHECK-NEXT: ldr     x28, [sp, #16]
; CHECK-NEXT: .seh_save_reg x28, 16
; CHECK-NEXT: ldp     x29, x30, [sp], #32
; CHECK-NEXT: .seh_save_fplr_x 32
; CHECK-NEXT: .seh_endepilogue
; CHECK-NEXT: ret


target datalayout = "e-m:w-p:64:64-i32:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64-unknown-windows-msvc19.11.0"

define dso_local void @"?a@@YAXXZ"() personality i8* bitcast (i32 (...)* @__CxxFrameHandler3 to i8*) {
entry:
  %a = alloca [100 x i32], align 64
  %0 = bitcast [100 x i32]* %a to i8*  
  call void @llvm.memset.p0i8.i64(i8* nonnull align 64 %0, i8 0, i64 400, i1 false)
  %1 = getelementptr inbounds [100 x i32], [100 x i32]* %a, i64 0, i64 0
  store i32 305419896, i32* %1, align 64
  invoke void @"?bb@@YAXPEAHH@Z"(i32* nonnull %1, i32 1)
          to label %try.cont unwind label %catch.dispatch

catch.dispatch:                                   ; preds = %entry
  %2 = catchswitch within none [label %catch] unwind to caller

catch:                                            ; preds = %catch.dispatch
  %3 = catchpad within %2 [i8* null, i32 64, i8* null]
  call void @"?bb@@YAXPEAHH@Z"(i32* nonnull %1, i32 0) [ "funclet"(token %3) ]
  catchret from %3 to label %try.cont

try.cont:                                         ; preds = %entry, %catch
  call void @"?cc@@YAXXZ"()
  ret void
}

declare void @llvm.memset.p0i8.i64(i8* nocapture writeonly, i8, i64, i1)

declare dso_local void @"?bb@@YAXPEAHH@Z"(i32*, i32)

declare dso_local i32 @__CxxFrameHandler3(...)

declare dso_local void @"?cc@@YAXXZ"()

