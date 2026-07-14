; RUN: llc -mtriple=aarch64-unknown-linux-gnu < %s | FileCheck %s
; RUN: llc -mtriple=aarch64-unknown-linux-gnu -global-isel < %s | FileCheck %s

define i64 @direct_rm_output() {
; CHECK-LABEL: direct_rm_output:
; CHECK:       // %bb.0:
; CHECK-NEXT:    //APP
; CHECK-NEXT:    mov x0, #42 // =0x2a
; CHECK-NEXT:    //NO_APP
; CHECK-NEXT:    ret
  %v = call i64 asm "mov $0, #42", "=rm"()
  ret i64 %v
}

define i64 @direct_rm_output_used(i64 %x) {
; CHECK-LABEL: direct_rm_output_used:
; CHECK:       // %bb.0:
; CHECK-NEXT:    //APP
; CHECK-NEXT:    mov x8, #42 // =0x2a
; CHECK-NEXT:    //NO_APP
; CHECK-NEXT:    add x0, x8, x0
; CHECK-NEXT:    ret
  %v = call i64 asm "mov $0, #42", "=rm"()
  %s = add i64 %v, %x
  ret i64 %s
}

define i64 @direct_rm_output_tied(i64 %x) {
; CHECK-LABEL: direct_rm_output_tied:
; CHECK:       // %bb.0:
; CHECK-NEXT:    //APP
; CHECK-NEXT:    add x0, x0, #1
; CHECK-NEXT:    //NO_APP
; CHECK-NEXT:    ret
  %v = call i64 asm "add $0, $0, #1", "=rm,0"(i64 %x)
  ret i64 %v
}

define void @indirect_m_output(ptr %p) {
; CHECK-LABEL: indirect_m_output:
; CHECK:       // %bb.0:
; CHECK-NEXT:    //APP
; CHECK-NEXT:    str xzr, [x0]
; CHECK-NEXT:    //NO_APP
; CHECK-NEXT:    ret
  call void asm "str xzr, $0", "=*m"(ptr elementtype(i64) %p)
  ret void
}
