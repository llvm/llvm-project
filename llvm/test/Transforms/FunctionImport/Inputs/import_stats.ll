; ModuleID = 'import_stats2.ll'
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@globalvar = global i32 1, align 4

define void @hot() {
  store i32 0, ptr @globalvar, align 4
  ret void
}
define void @critical() {
  ret void
}
define void @none() {
  ret void
}
