; RUN: llc -mattr=+sme -stop-after=finalize-isel < %s | FileCheck %s

target triple = "aarch64"

; Check that we don't define VG for 'smstart za' and 'smstop za'
define void @smstart_za() "aarch64_new_za" nounwind {
  ; CHECK-LABEL:    name: smstart_za
  ; CHECK-NOT:        implicit-def {{[^,]*}}$vg
  ret void
}

; Check that we do define VG for 'smstart sm' and 'smstop sm'
define void @smstart_sm() nounwind {
  ; CHECK-LABEL: name: smstart_sm
  ; CHECK:          MSRpstatesvcrImm1 1, 1,
  ; CHECK-SAME:       implicit-def {{[^,]*}}$vg
  ; CHECK:          MSRpstatesvcrImm1 1, 0,
  ; CHECK-SAME:       implicit-def {{[^,]*}}$vg
  call void @require_sm()
  ret void
}

declare void @require_sm() "aarch64_pstate_sm_enabled"
declare void @require_za() "aarch64_inout_za"
