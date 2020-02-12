; RUN: not llc -march=sparc < %s -o /dev/null 2>&1 | FileCheck %s

; CHECK: sparc only supports sret on the first parameter

define void @foo(i32 %a, i32* sret %out) {
  store i32 %a, i32* %out
  ret void
}
