; RUN: llc -mtriple=x86_64-unknown-linux-gnu -global-isel=0 < %s | FileCheck %s

define i64 @direct_rm_output() {
; CHECK-LABEL: direct_rm_output:
; CHECK:       # %bb.0:
; CHECK-NEXT:    #APP
; CHECK-NEXT:    movq $42, %rax
; CHECK-NEXT:    #NO_APP
; CHECK-NEXT:    retq
  %v = call i64 asm "movq $$42, $0", "=rm"()
  ret i64 %v
}

define i64 @direct_rm_output_used(i64 %x) {
; CHECK-LABEL: direct_rm_output_used:
; CHECK:       # %bb.0:
; CHECK-NEXT:    #APP
; CHECK-NEXT:    movq $42, %rax
; CHECK-NEXT:    #NO_APP
; CHECK-NEXT:    addq %rdi, %rax
; CHECK-NEXT:    retq
  %v = call i64 asm "movq $$42, $0", "=rm"()
  %s = add i64 %v, %x
  ret i64 %s
}

define void @indirect_m_output(ptr %p) {
; CHECK-LABEL: indirect_m_output:
; CHECK:       # %bb.0:
; CHECK-NEXT:    #APP
; CHECK-NEXT:    movq $0, (%rdi)
; CHECK-NEXT:    #NO_APP
; CHECK-NEXT:    retq
  call void asm "movq $$0, $0", "=*m"(ptr elementtype(i64) %p)
  ret void
}
