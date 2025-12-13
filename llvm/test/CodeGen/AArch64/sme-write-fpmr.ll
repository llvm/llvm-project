; RUN: llc -mattr=+sme -stop-after=finalize-isel < %s | FileCheck %s

target triple = "aarch64"

; Check that we don't define FPMR for 'smstart za' and 'smstop za'
define void @smstart_za() "aarch64_new_za" nounwind {
  ; CHECK-LABEL:    name: smstart_za
  ; CHECK-NOT:        implicit-def {{[^,]*}}$fpmr
  ret void
}

; Check that we do define FPMR for 'smstart sm' and 'smstop sm'
define void @smstart_sm() nounwind {
  ; CHECK-LABEL: name: smstart_sm
  ; CHECK:          MSRpstatesvcrImm1 1, 1,
  ; CHECK-SAME:       implicit-def {{[^,]*}}$fpmr
  ; CHECK:          MSRpstatesvcrImm1 1, 0,
  ; CHECK-SAME:       implicit-def {{[^,]*}}$fpmr
  call void @require_sm()
  ret void
}

declare void @require_sm() "aarch64_pstate_sm_enabled"
