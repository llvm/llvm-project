; RUN: llc -mtriple=aarch64-linux-gnu -mattr=+sme -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=aarch64-linux-gnu -mattr=+sme -verify-machineinstrs -stop-after=finalize-isel < %s | FileCheck %s --check-prefix=CHECK-CSRMASK

define i64 @get_pstatesm_normal() nounwind {
; CHECK-LABEL: get_pstatesm_normal:
; CHECK:       // %bb.0:
; CHECK-NEXT:    mov x0, xzr
; CHECK-NEXT:    ret
  %pstate = call i64 @llvm.aarch64.sme.get.pstatesm()
  ret i64 %pstate
}

define i64 @get_pstatesm_streaming() nounwind "aarch64_pstate_sm_enabled" {
; CHECK-LABEL: get_pstatesm_streaming:
; CHECK:       // %bb.0:
; CHECK-NEXT:    mov w0, #1
; CHECK-NEXT:    ret
  %pstate = call i64 @llvm.aarch64.sme.get.pstatesm()
  ret i64 %pstate
}

define i64 @get_pstatesm_locally_streaming() nounwind "aarch64_pstate_sm_body" {
; CHECK-LABEL: get_pstatesm_locally_streaming:
; CHECK:       // %bb.0:
; CHECK-NEXT:    mov w0, #1
; CHECK-NEXT:    ret
  %pstate = call i64 @llvm.aarch64.sme.get.pstatesm()
  ret i64 %pstate
}

define i64 @get_pstatesm_streaming_compatible() nounwind "aarch64_pstate_sm_compatible" {
; CHECK-LABEL: get_pstatesm_streaming_compatible:
; CHECK:       // %bb.0:
; CHECK-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; CHECK-NEXT:    bl __arm_sme_state
; CHECK-NEXT:    and x0, x0, #0x1
; CHECK-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; CHECK-NEXT:    ret
;
; CHECK-CSRMASK-LABEL: name: get_pstatesm_streaming_compatible
; CHECK-CSRMASK: BL &__arm_sme_state, csr_aarch64_sme_abi_support_routines_preservemost_from_x2
  %pstate = call i64 @llvm.aarch64.sme.get.pstatesm()
  ret i64 %pstate
}

declare i64 @llvm.aarch64.sme.get.pstatesm()
