target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-grtev4-linux-gnu"

@linkonceodrunnamed = linkonce_odr unnamed_addr constant i32 0
@odrunnamed = weak_odr unnamed_addr constant i32 0

; ld.bfd requires relocations to be present to count as a native code
; reference.
define void @refs() {
  load volatile i32, ptr @linkonceodrunnamed
  load volatile i32, ptr @odrunnamed
  ret void
}
