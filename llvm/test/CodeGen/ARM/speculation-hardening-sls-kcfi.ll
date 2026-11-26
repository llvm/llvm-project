; RUN: llc -verify-machineinstrs -mtriple=armv8-linux-gnueabi -O2 < %s | FileCheck %s --check-prefixes=CHECK,ARM
; RUN: llc -verify-machineinstrs -mtriple=thumbv8-linux-gnueabi -O2 < %s | FileCheck %s --check-prefixes=CHECK,THUMB

; Verify that ARM SLS BLR hardening lowers indirect calls inside KCFI bundles
; to SLS thunk calls instead of leaving unhardened blx instructions.

define void @call_kcfi(ptr %fn) #0 {
; CHECK-LABEL: call_kcfi:
; CHECK:       bic r12, r0, #1
; CHECK:       ldr r12, [r12, #-4]
; ARM:         udf #33760
; THUMB:       udf #128
; ARM:         bl __llvm_slsblr_thunk_arm_r0
; THUMB:       bl __llvm_slsblr_thunk_thumb_r0
; CHECK-NOT:   blx r0
entry:
  call void %fn() [ "kcfi"(i32 12345678) ]
  ret void
}

define void @tailcall_kcfi(ptr %fn) #0 {
; CHECK-LABEL: tailcall_kcfi:
; CHECK:       bic r12, r0, #1
; CHECK:       ldr r12, [r12, #-4]
; CHECK:       bx r0
; CHECK-NEXT:  dsb sy
; CHECK-NEXT:  isb sy
entry:
  tail call void %fn() [ "kcfi"(i32 12345678) ]
  ret void
}

attributes #0 = { nounwind "target-features"="+harden-sls-blr,+harden-sls-retbr" }

!llvm.module.flags = !{!0}
!0 = !{i32 4, !"kcfi", i32 1}
