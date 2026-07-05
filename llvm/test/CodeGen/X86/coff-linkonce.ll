; RUN: llc -function-sections -o - %s | FileCheck %s

target datalayout = "e-m:w-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc"

; CHECK: .section{{.*}}one_only
define linkonce_odr void @foo() {
  ret void
}
