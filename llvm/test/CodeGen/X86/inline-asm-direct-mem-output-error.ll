; RUN: not llc -mtriple=x86_64-unknown-linux-gnu -global-isel=0 < %s 2>&1 | FileCheck %s

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

; CHECK: error: address constraint 'p' is only valid as an input
define i64 @direct_p_output() {
  %v = call i64 asm "", "=p"()
  ret i64 %v
}

; CHECK: error: address constraint 'p' is only valid as an input
define void @indirect_p_output(ptr %x) {
  call void asm "", "=*p"(ptr elementtype(i64) %x)
  ret void
}
