target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @callee(ptr %target) {
  call void %target(), !callees !0
  ret void
}

define internal void @metadata_only_target() {
  ret void
}

!0 = !{ptr @metadata_only_target, ptr null}
