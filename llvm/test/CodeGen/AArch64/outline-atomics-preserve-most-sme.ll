; RUN: llc -mtriple=aarch64-unknown-linux-gnu -mattr=+sme2,+outline-atomics -verify-machineinstrs < %s | FileCheck %s --check-prefix=DEFAULT
; RUN: llc -mtriple=aarch64-unknown-linux-gnu -mattr=+sme2,+outline-atomics,+outline-atomics-preserve-most -verify-machineinstrs < %s | FileCheck %s --check-prefix=PRESERVE

; Verify that __aarch64_*_preservemost helpers are treated as streaming-compatible
; and ZA/ZT0-agnostic, avoiding smstop/smstart and ZA/ZT0 saves in SME callers.
define i64 @ldadd8_sme_streaming_za_zt0(ptr %p, i64 %v) "aarch64_pstate_sm_enabled" "aarch64_inout_za" "aarch64_inout_zt0" nounwind {
; DEFAULT-LABEL: ldadd8_sme_streaming_za_zt0:
; DEFAULT:       smstop sm
; DEFAULT:       bl __aarch64_ldadd8_acq_rel{{$}}
; DEFAULT:       smstart sm
;
; PRESERVE-LABEL: ldadd8_sme_streaming_za_zt0:
; PRESERVE-NOT:   smstop
; PRESERVE-NOT:   __arm_tpidr2_save
; PRESERVE-NOT:   str zt0
; PRESERVE:       bl __aarch64_ldadd8_acq_rel_preservemost{{$}}
; PRESERVE-NOT:   smstart
; PRESERVE-NOT:   ldr zt0
; PRESERVE:       ret
  %old = atomicrmw add ptr %p, i64 %v acq_rel
  ret i64 %old
}
