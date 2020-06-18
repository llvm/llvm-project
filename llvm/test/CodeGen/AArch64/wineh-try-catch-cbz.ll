; RUN: llc < %s | FileCheck %s

; Make sure the prologue is sane.  (Doesn't need to exactly match this,
; but the original issue only reproduced if the cbz was immediately
; after the frame setup.)

; CHECK: stp     x29, x30, [sp, #-32]!
; CHECK-NEXT: mov     x29, sp
; CHECK-NEXT: mov     x1, #-2
; CHECK-NEXT: stur    x1, [x29, #16]
; CHECK-NEXT: cbz     w0, .LBB0_2

target datalayout = "e-m:w-p:64:64-i32:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64-unknown-windows-msvc19.11.0"

; Function Attrs: uwtable
define dso_local void @"?f@@YAXH@Z"(i32 %x) local_unnamed_addr #0 personality i8* bitcast (i32 (...)* @__CxxFrameHandler3 to i8*) {
entry:
  %cmp = icmp eq i32 %x, 0
  br i1 %cmp, label %try.cont, label %if.then

if.then:                                          ; preds = %entry
  invoke void @"?g@@YAXXZ"()
          to label %try.cont unwind label %catch.dispatch

catch.dispatch:                                   ; preds = %if.then
  %0 = catchswitch within none [label %catch] unwind to caller

catch:                                            ; preds = %catch.dispatch
  %1 = catchpad within %0 [i8* null, i32 64, i8* null]
  catchret from %1 to label %try.cont

try.cont:                                         ; preds = %entry, %if.then, %catch
  ret void
}

declare dso_local void @"?g@@YAXXZ"() local_unnamed_addr #1

declare dso_local i32 @__CxxFrameHandler3(...)
