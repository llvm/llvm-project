; RUN: not llc -mtriple=x86_64-unknown-unknown -stackrealign -verify-machineinstrs %s -o - 2>&1 | FileCheck %s

declare ghccc void @may_throw_or_crash()
declare i32 @_except_handler3(...)

define internal i64 @catchall_filt() {
  ret i64 1
}

; If the invoked function clobbers frame pointer and/or base pointer according
; to its calling convention, we can't handle it currently, so reports an error
; message.

; CHECK: <unknown>:0: error: Frame pointer clobbered by function invoke is not supported
; CHECK: <unknown>:0: error: Stack realignment in presence of dynamic allocas is not supported with this calling convention
define void @use_except_handler3() personality ptr @_except_handler3 {
entry:
  invoke ghccc void @may_throw_or_crash()
      to label %cont unwind label %lpad
cont:
  ret void
lpad:
  %cs = catchswitch within none [label %catch] unwind to caller
catch:
  %p = catchpad within %cs [ptr @catchall_filt]
  catchret from %p to label %cont
}
