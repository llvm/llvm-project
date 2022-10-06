; RUN: not llvm-as %s -o /dev/null 2>&1 | FileCheck %s

declare void @sm_attrs() "aarch64_pstate_sm_enabled" "aarch64_pstate_sm_compatible";
; CHECK: Attributes 'aarch64_pstate_sm_enabled and aarch64_pstate_sm_compatible' are incompatible!

declare void @za_preserved() "aarch64_pstate_za_new" "aarch64_pstate_za_preserved";
; CHECK: Attributes 'aarch64_pstate_za_new and aarch64_pstate_za_preserved' are incompatible!

declare void @za_shared() "aarch64_pstate_za_new" "aarch64_pstate_za_shared";
; CHECK: Attributes 'aarch64_pstate_za_new and aarch64_pstate_za_shared' are incompatible!
