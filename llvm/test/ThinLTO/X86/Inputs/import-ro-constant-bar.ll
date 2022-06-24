source_filename = "bar.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@foo = external dso_local local_unnamed_addr constant i32, align 4
define dso_local i32 @_Z3barv() local_unnamed_addr {
entry:
  %0 = load i32, ptr @foo, align 4
  ret i32 %0
}
