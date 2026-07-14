; RUN: not llc -mtriple=aarch64-unknown-linux-gnu -global-isel=0 < %s 2>&1 | FileCheck %s
; RUN: not llc -mtriple=aarch64-unknown-linux-gnu -global-isel=1 < %s 2>&1 | FileCheck %s

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
