; RUN: llc %s -o - -verify-machineinstrs | FileCheck %s

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64-arm-none-eabi"

@global0 = dso_local local_unnamed_addr global i32 0, align 4
@global1 = dso_local local_unnamed_addr global i32 0, align 4

define dso_local i32 @func() minsize optsize {
; CHECK-LABEL: @func
; CHECK:       adrp x8, .L_MergedGlobals
; CHECK-NEXT:  add x8, x8, :lo12:.L_MergedGlobals
; CHECK-NEXT:  ldp w9, w8, [x8]
; CHECK-NEXT:  add w0, w8, w9
; CHECK-NEXT:  ret
entry:
  %0 = load i32, i32* @global0, align 4
  %1 = load i32, i32* @global1, align 4
  %add = add nsw i32 %1, %0
  ret i32 %add
}
