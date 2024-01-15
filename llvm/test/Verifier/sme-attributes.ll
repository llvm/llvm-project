; RUN: not llvm-as %s -o /dev/null 2>&1 | FileCheck %s

declare void @sm_attrs() "aarch64_pstate_sm_enabled" "aarch64_pstate_sm_compatible";
; CHECK: Attributes 'aarch64_pstate_sm_enabled and aarch64_pstate_sm_compatible' are incompatible!

declare void @za_preserved() "aarch64_pstate_za_new" "aarch64_pstate_za_preserved";
; CHECK: Attributes 'aarch64_pstate_za_new and aarch64_pstate_za_preserved' are incompatible!

declare void @za_shared() "aarch64_pstate_za_new" "aarch64_pstate_za_shared";
; CHECK: Attributes 'aarch64_pstate_za_new and aarch64_pstate_za_shared' are incompatible!

declare void @zt0_new_preserved() "aarch64_sme_new_zt0" "aarch64_sme_preserved_zt0";
; CHECK: ZT0 state attributes 'aarch64_sme_[in|inout|out|new|preserved]_zt' are mutually exclusive!

declare void @zt0_new_in() "aarch64_sme_new_zt0" "aarch64_sme_in_zt0";
; CHECK: ZT0 state attributes 'aarch64_sme_[in|inout|out|new|preserved]_zt' are mutually exclusive!

declare void @zt0_new_inout() "aarch64_sme_new_zt0" "aarch64_sme_inout_zt0";
; CHECK: ZT0 state attributes 'aarch64_sme_[in|inout|out|new|preserved]_zt' are mutually exclusive!

declare void @zt0_new_out() "aarch64_sme_new_zt0" "aarch64_sme_out_zt0";
; CHECK: ZT0 state attributes 'aarch64_sme_[in|inout|out|new|preserved]_zt' are mutually exclusive!

declare void @zt0_preserved_in() "aarch64_sme_preserved_zt0" "aarch64_sme_in_zt0";
; CHECK: ZT0 state attributes 'aarch64_sme_[in|inout|out|new|preserved]_zt' are mutually exclusive!

declare void @zt0_preserved_inout() "aarch64_sme_preserved_zt0" "aarch64_sme_inout_zt0";
; CHECK: ZT0 state attributes 'aarch64_sme_[in|inout|out|new|preserved]_zt' are mutually exclusive!

declare void @zt0_preserved_out() "aarch64_sme_preserved_zt0" "aarch64_sme_out_zt0";
; CHECK: ZT0 state attributes 'aarch64_sme_[in|inout|out|new|preserved]_zt' are mutually exclusive!

declare void @zt0_in_inout() "aarch64_sme_in_zt0" "aarch64_sme_inout_zt0";
; CHECK: ZT0 state attributes 'aarch64_sme_[in|inout|out|new|preserved]_zt' are mutually exclusive!

declare void @zt0_in_out() "aarch64_sme_in_zt0" "aarch64_sme_out_zt0";
; CHECK: ZT0 state attributes 'aarch64_sme_[in|inout|out|new|preserved]_zt' are mutually exclusive!

declare void @zt0_inout_out() "aarch64_sme_inout_zt0" "aarch64_sme_out_zt0";
; CHECK: ZT0 state attributes 'aarch64_sme_[in|inout|out|new|preserved]_zt' are mutually exclusive!
