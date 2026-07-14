; RUN: not llc -mtriple=x86_64-unknown-linux-gnu < %s 2>&1 | FileCheck %s

; CHECK: error: memory output constraint 'm' must be indirect
define i64 @direct_m_output() {
  %v = call i64 asm "", "=m"()
  ret i64 %v
}

; CHECK: error: memory output constraint '{{[mo]}}' must be indirect
define i64 @direct_mo_output() {
  %v = call i64 asm "", "=mo"()
  ret i64 %v
}
