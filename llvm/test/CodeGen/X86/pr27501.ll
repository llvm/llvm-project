; RUN: llc < %s | FileCheck %s
target datalayout = "e-m:w-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc"

define void @test1(i64* %result.repack) personality i32 (...)* @__CxxFrameHandler3 {
bb:
  invoke void @may_throw(i32 1)
          to label %postinvoke unwind label %cleanuppad
; CHECK:         movq    %rcx, [[SpillLoc:.*\(%rbp\)]]
; CHECK:        movl    $1, %ecx
; CHECK:        callq   may_throw

postinvoke:                                       ; preds = %bb
  store i64 19, i64* %result.repack, align 8
; CHECK:        movq	[[SpillLoc]], [[R1:%r..]]
; CHECK:        movq    $19, ([[R1]])
; CHECK:        movl    $2, %ecx
; CHECK-NEXT:   callq   may_throw
  invoke void @may_throw(i32 2)
          to label %assertFailed unwind label %catch.dispatch

catch.dispatch:                                   ; preds = %cleanuppad9, %postinvoke
  %tmp3 = catchswitch within none [label %catch.object.Throwable] unwind label %cleanuppad

catch.object.Throwable:                           ; preds = %catch.dispatch
  %tmp2 = catchpad within %tmp3 [i8* null, i32 64, i8* null]
  catchret from %tmp2 to label %catchhandler

catchhandler:                                     ; preds = %catch.object.Throwable
  invoke void @may_throw(i32 3)
          to label %try.success.or.caught unwind label %cleanuppad

try.success.or.caught:                            ; preds = %catchhandler
  invoke void @may_throw(i32 4)
          to label %postinvoke27 unwind label %cleanuppad24
; CHECK:        movl    $4, %ecx
; CHECK-NEXT:   callq   may_throw

postinvoke27:                                     ; preds = %try.success.or.caught
  store i64 42, i64* %result.repack, align 8
; CHECK:        movq    [[SpillLoc]], [[R2:%r..]]
; CHECK-NEXT:   movq    $42, ([[R2]])
  ret void

cleanuppad24:                                     ; preds = %try.success.or.caught
  %tmp5 = cleanuppad within none []
  cleanupret from %tmp5 unwind to caller

cleanuppad:                                       ; preds = %catchhandler, %catch.dispatch, %bb
  %tmp1 = cleanuppad within none []
  cleanupret from %tmp1 unwind to caller

assertFailed:                                     ; preds = %postinvoke
  invoke void @may_throw(i32 5)
          to label %postinvoke13 unwind label %cleanuppad9

postinvoke13:                                     ; preds = %assertFailed
  unreachable

cleanuppad9:                                      ; preds = %assertFailed
  %tmp4 = cleanuppad within none []
  cleanupret from %tmp4 unwind label %catch.dispatch
}

declare void @may_throw(i32)

declare i32 @__CxxFrameHandler3(...)
