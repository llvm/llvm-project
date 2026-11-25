; RUN: llc -verify-machineinstrs -mtriple=aarch64-linux-gnu -O2 < %s | FileCheck %s
; RUN: llc -verify-machineinstrs -mtriple=aarch64-linux-gnu -global-isel -O2 < %s | FileCheck %s

; Verify that combining KCFI with SLS BLR hardening lowers cleanly through
; AArch64AsmPrinter::LowerKCFI_CHECK when AArch64SLSHardening converts BLRNoIP
; inside the KCFI bundle into a direct BL to the SLS thunk.

define void @call_kcfi(ptr %fn) #0 {
; CHECK-LABEL: call_kcfi:
; CHECK:       ldur w16, [x0, #-4]
; CHECK:       brk #0x8220
; CHECK:       bl __llvm_slsblr_thunk_x0
; CHECK-NOT:   blr x0
entry:
  call void %fn() [ "kcfi"(i32 12345678) ]
  ret void
}

attributes #0 = { nounwind "target-features"="+harden-sls-blr,+harden-sls-retbr" }

!llvm.module.flags = !{!0}
!0 = !{i32 4, !"kcfi", i32 1}
