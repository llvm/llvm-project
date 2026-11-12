target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

module asm ".symver bar, bar@BAR_1.2.3"

declare dso_local i32 @bar()

define dso_local i32 @foo() {
entry:
  %call = tail call i32 @bar()
  ret i32 %call
}
