target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

source_filename = "thinlto_indirect_call_promotion.c"

define void @a() {
entry:
  ret void
}

define internal void @c() !PGOFuncName !1 {
entry:
  ret void
}

!1 = !{!"thinlto_indirect_call_promotion.c:c"}
