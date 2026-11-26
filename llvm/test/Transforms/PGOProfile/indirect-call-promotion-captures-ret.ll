; RUN: opt < %s -S -passes=pgo-icall-prom -pass-remarks-missed=pgo-icall-prom 2>&1 | FileCheck %s

; CHECK: remark: <unknown>:0:0: Cannot promote indirect call to target (count=1600): captures(ret: ...) with non-pointer return type

define i64 @target(ptr %p) {
  ret i64 0
}

define ptr @caller(ptr %fp, ptr %p) {
  %r = call ptr %fp(ptr captures(ret: address, provenance) %p), !prof !0
  ret ptr %r
}

!0 = !{!"VP", i32 0, i64 1600, i64 -2747246343000183230, i64 1600}
