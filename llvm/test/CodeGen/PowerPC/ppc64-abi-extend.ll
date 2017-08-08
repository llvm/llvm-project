; Verify that i32 argument/return values are extended to i64

; RUN: llc -verify-machineinstrs < %s | FileCheck %s
target datalayout = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-f128:128:128-v128:128:128-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

@si = common global i32 0, align 4
@ui = common global i32 0, align 4

declare void @arg_si(i32 signext)
declare void @arg_ui(i32 zeroext)

declare signext i32 @ret_si()
declare zeroext i32 @ret_ui()

define void @pass_arg_si() nounwind {
entry:
  %0 = load i32, i32* @si, align 4
  tail call void @arg_si(i32 signext %0) nounwind
  ret void
}
; CHECK: @pass_arg_si
; CHECK: lwa 3,
; CHECK: bl arg_si

define void @pass_arg_ui() nounwind {
entry:
  %0 = load i32, i32* @ui, align 4
  tail call void @arg_ui(i32 zeroext %0) nounwind
  ret void
}
; CHECK: @pass_arg_ui
; CHECK: lwz 3,
; CHECK: bl arg_ui

define i64 @use_arg_si(i32 signext %x) nounwind readnone {
entry:
  %conv = sext i32 %x to i64
  ret i64 %conv
}
; CHECK: @use_arg_si
; CHECK: %entry
; CHECK-NEXT: blr

define i64 @use_arg_ui(i32 zeroext %x) nounwind readnone {
entry:
  %conv = zext i32 %x to i64
  ret i64 %conv
}
; CHECK: @use_arg_ui
; CHECK: %entry
; CHECK-NEXT: blr

define signext i32 @pass_ret_si() nounwind readonly {
entry:
  %0 = load i32, i32* @si, align 4
  ret i32 %0
}
; CHECK: @pass_ret_si
; CHECK: lwa 3,
; CHECK: blr

define zeroext i32 @pass_ret_ui() nounwind readonly {
entry:
  %0 = load i32, i32* @ui, align 4
  ret i32 %0
}
; CHECK: @pass_ret_ui
; CHECK: lwz 3,
; CHECK: blr

define i64 @use_ret_si() nounwind {
entry:
  %call = tail call signext i32 @ret_si() nounwind
  %conv = sext i32 %call to i64
  ret i64 %conv
}
; CHECK: @use_ret_si
; CHECK: bl ret_si
; This is to verify the return register (3) set up by the ret_si
; call is passed on unmodified as return value of use_ret_si.
; CHECK-NOT: 3
; CHECK: blr

define i64 @use_ret_ui() nounwind {
entry:
  %call = tail call zeroext i32 @ret_ui() nounwind
  %conv = zext i32 %call to i64
  ret i64 %conv
}
; CHECK: @use_ret_ui
; CHECK: bl ret_ui
; This is to verify the return register (3) set up by the ret_ui
; call is passed on unmodified as return value of use_ret_ui.
; CHECK-NOT: 3
; CHECK: blr

