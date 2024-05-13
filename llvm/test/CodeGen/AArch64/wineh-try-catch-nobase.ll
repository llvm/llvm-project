; RUN: llc -o - %s -mtriple=aarch64-windows -verify-machineinstrs | FileCheck %s

; Make sure we don't have a base pointer.
; CHECK-LABEL: "?a@@YAXXZ":
; CHECK-NOT: x19

; Check that we compute the address relative to fp.
; CHECK-LABEL: "?catch$2@?0??a@@YAXXZ@4HA":
; CHECK:             stp     x29, x30, [sp, #-16]!   // 16-byte Folded Spill
; CHECK-NEXT:        .seh_save_fplr_x 16
; CHECK-NEXT:        .seh_endprologue
; CHECK-NEXT:        sub     x0, x29, #16
; CHECK-NEXT:        mov     x1, xzr
; CHECK-NEXT:        bl      "?bb@@YAXPEAHH@Z"
; CHECK-NEXT:        adrp    x0, .LBB0_1
; CHECK-NEXT:        add     x0, x0, .LBB0_1
; CHECK-NEXT:        .seh_startepilogue
; CHECK-NEXT:        ldp     x29, x30, [sp], #16     // 16-byte Folded Reload
; CHECK-NEXT:        .seh_save_fplr_x 16
; CHECK-NEXT:        .seh_endepilogue
; CHECK-NEXT:        ret

target datalayout = "e-m:w-p:64:64-i32:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64-unknown-windows-msvc19.11.0"

define dso_local void @"?a@@YAXXZ"(i64 %p1) personality ptr @__CxxFrameHandler3 {
entry:
  %a = alloca i32, align 16
  store i32 305419896, ptr %a, align 16
  invoke void @"?bb@@YAXPEAHH@Z"(ptr nonnull %a, ptr null)
          to label %try.cont unwind label %catch.dispatch

catch.dispatch:                                   ; preds = %entry
  %0 = catchswitch within none [label %catch] unwind to caller

catch:                                            ; preds = %catch.dispatch
  %1 = catchpad within %0 [ptr null, i32 64, ptr null]
  call void @"?bb@@YAXPEAHH@Z"(ptr nonnull %a, ptr null) [ "funclet"(token %1) ]
  catchret from %1 to label %try.cont

try.cont:                                         ; preds = %entry, %catch
  call void @"?cc@@YAXXZ"()
  ret void
}

declare void @llvm.memset.p0.i64(ptr nocapture writeonly, i8, i64, i1)

declare dso_local void @"?bb@@YAXPEAHH@Z"(ptr, ptr)

declare dso_local i32 @__CxxFrameHandler3(...)

declare dso_local void @"?cc@@YAXXZ"()

