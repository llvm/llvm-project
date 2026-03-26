; RUN: llc -mtriple=thumbv8.1m.main %s -o - | FileCheck %s

@foo = unnamed_addr alias void (), ptr @bar

; CHECK: .globl bar 
; CHECK: .globl __acle_se_bar @ @bar 
; CHECK: .globl foo 
; CHECK: foo = bar
; CHECK: __acle_se_foo = __acle_se_bar

define dso_local void @bar() unnamed_addr #0 {
start:
  ret void
}

attributes #0 = { nounwind "cmse_nonsecure_entry" }

