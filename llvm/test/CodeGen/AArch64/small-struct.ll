; RUN: llc -filetype=asm -O3 -asm-verbose=false %s -o - | FileCheck %s
; CHECK:      main:
; CHECK-NEXT: .cfi_startproc
; CHECK-NEXT: str x30, [sp, #-16]!
; CHECK-NEXT: .cfi_def_cfa_offset 16
; CHECK-NEXT: .cfi_offset w30, -16
; CHECK-NEXT: bl _Z5getU4v
; CHECK-NEXT: ldr x30, [sp], #16
; CHECK-NEXT: b _Z3bar2U4


target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-linux-gnu"

; Function Attrs: mustprogress norecurse uwtable(sync)
define dso_local noundef i32 @main() local_unnamed_addr {
entry:
  %call = tail call i32 @_Z5getU4v()
  %coerce.val.ii = zext i32 %call to i64
  %call2 = tail call noundef i32 @_Z3bar2U4(i64 noext "bitwidth"="32" %coerce.val.ii)
  ret i32 %call2
}

declare dso_local noext i32 @_Z5getU4v() local_unnamed_addr

declare dso_local noundef i32 @_Z3bar2U4(i64 noext) local_unnamed_addr

