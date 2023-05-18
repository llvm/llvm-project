; RUN: llc -mtriple=aarch64-linux-gnu -mattr=+sme -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=aarch64-linux-gnu -mattr=+sme -verify-machineinstrs -stop-after=finalize-isel < %s | FileCheck %s --check-prefix=CHECK-CSRMASK

; Test that the PCS attribute is accepted and uses the correct register mask.
;

define void @test_sme_calling_convention_x0() nounwind {
; CHECK-LABEL: test_sme_calling_convention_x0:
; CHECK:       // %bb.0:
; CHECK-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; CHECK-NEXT:    bl __arm_tpidr2_save
; CHECK-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; CHECK-NEXT:    ret
;
; CHECK-CSRMASK-LABEL: name: test_sme_calling_convention_x0
; CHECK-CSRMASK: BL @__arm_tpidr2_save, csr_aarch64_sme_abi_support_routines_preservemost_from_x0
  call aarch64_sme_preservemost_from_x0 void @__arm_tpidr2_save()
  ret void
}

define i64 @test_sme_calling_convention_x2() nounwind {
; CHECK-LABEL: test_sme_calling_convention_x2:
; CHECK:       // %bb.0:
; CHECK-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; CHECK-NEXT:    bl __arm_sme_state
; CHECK-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; CHECK-NEXT:    ret
;
; CHECK-CSRMASK-LABEL: name: test_sme_calling_convention_x2
; CHECK-CSRMASK: BL @__arm_sme_state, csr_aarch64_sme_abi_support_routines_preservemost_from_x2
  %pstate = call aarch64_sme_preservemost_from_x2 {i64, i64} @__arm_sme_state()
  %pstate.sm = extractvalue {i64, i64} %pstate, 0
  ret i64 %pstate.sm
}

declare void @__arm_tpidr2_save()
declare {i64, i64} @__arm_sme_state()
