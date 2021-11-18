; RUN: opt < %s -passes=globalopt -S | FileCheck %s

; CHECK: ModuleID
define internal i32 @__cxa_atexit(void (i8*)* nocapture %func, i8* nocapture %arg, i8* nocapture %dso_handle) nounwind readnone optsize noimplicitfloat {
  unreachable
}
