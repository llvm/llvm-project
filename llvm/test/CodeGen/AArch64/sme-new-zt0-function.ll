; RUN: opt -S -mtriple=aarch64-linux-gnu -aarch64-sme-abi %s | FileCheck %s

declare void @callee();

define void @private_za() "aarch64_new_zt0" {
  call void @callee()
  ret void
}

; CHECK: call aarch64_sme_preservemost_from_x0 void @__arm_tpidr2_save() #[[TPIDR2_SAVE_CALL_ATTR:[0-9]+]]
; CHECK: declare void @__arm_tpidr2_save() #[[TPIDR2_SAVE_DECL_ATTR:[0-9]+]]

; CHECK: attributes #[[TPIDR2_SAVE_DECL_ATTR]] = { "aarch64_pstate_sm_compatible" }
; CHECK: attributes #[[TPIDR2_SAVE_CALL_ATTR]] = { "aarch64_zt0_undef" }
