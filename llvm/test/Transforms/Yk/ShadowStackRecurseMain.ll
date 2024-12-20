; Checks that a recurive call to main crashes the shadow stack pass.
;
; RUN: not llc -O0 -stop-after yk-shadow-stack-pass -yk-shadow-stack < %s 2>&1 | FileCheck %s

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"


; CHECK: error: detected recursive call to main!
define dso_local i32 @main(i32 noundef %argc, ptr noundef %argv) noinline optnone {
entry:
  %rv = call i32 @main(i32 %argc, ptr %argv);
  ret i32 %rv
}
