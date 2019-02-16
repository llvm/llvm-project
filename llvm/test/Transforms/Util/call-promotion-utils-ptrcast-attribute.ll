; RUN: opt -S -pgo-icall-prom -icp-total-percent-threshold=0 < %s 2>&1 | FileCheck %s

; Test that CallPromotionUtils will promote calls which require pointer cast
; safely, i.e. drop incompatible attributes.

@foo = common global i8* (i8*)* null, align 8

; casting to i64 and pointer attribute at callsite dropped.
define i64 @func2(i64 %a) {
  ret i64 undef
}

; no casting needed, attribute at callsite preserved.
define i8* @func4(i8* %a) {
  ret i8* undef
}

define i8* @bar(i8* %arg) {
  %tmp = load i8* (i8*)*, i8* (i8*)** @foo, align 8

; Make sure callsite attributes are preserved on arguments and retval.
; CHECK: call noalias i8* @func4(i8* nonnull

; Make sure callsite attributes are dropped on arguments and retval.
; CHECK: [[ARG:%[0-9]+]] = ptrtoint i8* %arg to i64
; CHECK-NEXT: call i64 @func2(i64 [[ARG]])

  %call = call noalias i8* %tmp(i8* nonnull %arg), !prof !1
  ret i8* %call
}

!1 = !{!"VP", i32 0, i64 1440, i64 7651369219802541373, i64 1030, i64 -4377547752858689819, i64 410}
