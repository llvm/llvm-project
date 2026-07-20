; RUN: llc -mtriple=arm64-apple-macosx -verify-machineinstrs < %s | FileCheck %s

; Verify that the stack-probing attribute emitted by Apple Swift is accepted
; on Darwin and lowered to the existing inline stack-probing sequence.

define void @swift_darwin_stack_probe(ptr %out) #0 {
; CHECK-LABEL: swift_darwin_stack_probe:
; CHECK:       sub sp, sp, #1, lsl #12
; CHECK-NEXT:  ldr xzr, [sp]
entry:
  %frame = alloca i8, i64 4096, align 1
  store ptr %frame, ptr %out, align 8
  ret void
}

attributes #0 = {
  noinline
  "frame-pointer"="none"
  "probe-stack"="__chkstk_darwin"
  "stack-probe-size"="4096"
  uwtable(async)
}
