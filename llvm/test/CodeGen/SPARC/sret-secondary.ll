; RUN: not --crash llc -mtriple=sparc < %s -o /dev/null 2>&1 | FileCheck %s

; CHECK: sparc only supports sret on the first parameter

define void @foo(i32 %a, ptr sret(i32) %out) {
  store i32 %a, ptr %out
  ret void
}
