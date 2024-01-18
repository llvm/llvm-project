; RUN: llc --stop-after=finalize-isel -o - | FileCheck %s
target triple = "aarch64-linux"

; Check dynamic stack allocation and probing instructions do not have
; the FrameSetup flag.

; CHECK-NOT: frame-setup
define void @no_frame_setup(i64 %size, ptr %out) #0 {
  %v = alloca i8, i64 %size, align 1
  store ptr %v, ptr %out, align 8
  ret void
}

attributes #0 = { uwtable(async) "probe-stack"="inline-asm" "frame-pointer"="none" }
