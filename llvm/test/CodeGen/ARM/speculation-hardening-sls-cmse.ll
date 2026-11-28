; RUN: llc -verify-machineinstrs -mtriple=thumbv8m.main-none-eabi -O2 < %s | FileCheck %s --check-prefixes=CHECK,ISBDSB
; RUN: llc -verify-machineinstrs -mtriple=thumbv8m.main-none-eabi -mattr=+sb -O2 < %s | FileCheck %s --check-prefixes=CHECK,SB

; Verify that ARM SLS hardening inserts a speculation barrier after CMSE
; non-secure entry returns (bxns) and does not crash on CMSE non-secure calls
; (blxns).

define i32 @secure_entry(i32 %x) #0 {
; CHECK-LABEL: secure_entry:
; CHECK:       bxns lr
; ISBDSB-NEXT: dsb sy
; ISBDSB-NEXT: isb sy
; SB-NEXT:     sb
entry:
  %add = add i32 %x, 1
  ret i32 %add
}

define void @call_ns(ptr %fn) #1 {
; CHECK-LABEL: call_ns:
; CHECK:       blxns r0
; CHECK:       pop {r7, pc}
; ISBDSB-NEXT: dsb sy
; ISBDSB-NEXT: isb sy
; SB-NEXT:     sb
entry:
  call void %fn() "cmse_nonsecure_call"
  ret void
}

attributes #0 = { nounwind "cmse_nonsecure_entry" "target-features"="+8msecext,+harden-sls-retbr,+harden-sls-blr" }
attributes #1 = { nounwind "target-features"="+8msecext,+harden-sls-retbr,+harden-sls-blr" }
