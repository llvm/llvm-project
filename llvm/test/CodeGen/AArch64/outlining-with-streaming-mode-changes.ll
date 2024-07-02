; RUN: llc -mtriple=aarch64-linux-gnu -mattr=+sve -mattr=+sme2 -enable-machine-outliner -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=aarch64-linux-gnu -mattr=+sve -mattr=+sme2 -enable-machine-outliner -verify-machineinstrs < %s | FileCheck %s -check-prefix=OUTLINER

declare void @callee();

define void @streaming_mode_change1() #0 {
; CHECK-LABEL: streaming_mode_change1:
; CHECK:       // %bb.0:
; CHECK-NEXT:    stp d15, d14, [sp, #-80]! // 16-byte Folded Spill
; CHECK-NEXT:    cntd x9
; CHECK-NEXT:    stp d13, d12, [sp, #16] // 16-byte Folded Spill
; CHECK-NEXT:    stp d11, d10, [sp, #32] // 16-byte Folded Spill
; CHECK-NEXT:    stp d9, d8, [sp, #48] // 16-byte Folded Spill
; CHECK-NEXT:    stp x30, x9, [sp, #64] // 16-byte Folded Spill
; CHECK-NEXT:    smstop sm
; CHECK-NEXT:    bl callee
; CHECK-NEXT:    smstart sm
; CHECK-NEXT:    ldp d9, d8, [sp, #48] // 16-byte Folded Reload
; CHECK-NEXT:    ldr x30, [sp, #64] // 8-byte Folded Reload
; CHECK-NEXT:    ldp d11, d10, [sp, #32] // 16-byte Folded Reload
; CHECK-NEXT:    ldp d13, d12, [sp, #16] // 16-byte Folded Reload
; CHECK-NEXT:    ldp d15, d14, [sp], #80 // 16-byte Folded Reload
; CHECK-NEXT:    ret
;
; OUTLINER-LABEL: streaming_mode_change1:
; OUTLINER-NOT: OUTLINED_FUNCTION_
;
  call void @callee();
  ret void;
}

define void @streaming_mode_change2() #0 {
; CHECK-LABEL: streaming_mode_change2:
; CHECK:       // %bb.0:
; CHECK-NEXT:    stp d15, d14, [sp, #-80]! // 16-byte Folded Spill
; CHECK-NEXT:    cntd x9
; CHECK-NEXT:    stp d13, d12, [sp, #16] // 16-byte Folded Spill
; CHECK-NEXT:    stp d11, d10, [sp, #32] // 16-byte Folded Spill
; CHECK-NEXT:    stp d9, d8, [sp, #48] // 16-byte Folded Spill
; CHECK-NEXT:    stp x30, x9, [sp, #64] // 16-byte Folded Spill
; CHECK-NEXT:    smstop sm
; CHECK-NEXT:    bl callee
; CHECK-NEXT:    smstart sm
; CHECK-NEXT:    ldp d9, d8, [sp, #48] // 16-byte Folded Reload
; CHECK-NEXT:    ldr x30, [sp, #64] // 8-byte Folded Reload
; CHECK-NEXT:    ldp d11, d10, [sp, #32] // 16-byte Folded Reload
; CHECK-NEXT:    ldp d13, d12, [sp, #16] // 16-byte Folded Reload
; CHECK-NEXT:    ldp d15, d14, [sp], #80 // 16-byte Folded Reload
; CHECK-NEXT:    ret
;
; OUTLINER-LABEL: streaming_mode_change2:
; OUTLINER-NOT: OUTLINED_FUNCTION_
;
  call void @callee();
  ret void;
}

define void @streaming_mode_change3() #0 {
; CHECK-LABEL: streaming_mode_change3:
; CHECK:       // %bb.0:
; CHECK-NEXT:    stp d15, d14, [sp, #-80]! // 16-byte Folded Spill
; CHECK-NEXT:    cntd x9
; CHECK-NEXT:    stp d13, d12, [sp, #16] // 16-byte Folded Spill
; CHECK-NEXT:    stp d11, d10, [sp, #32] // 16-byte Folded Spill
; CHECK-NEXT:    stp d9, d8, [sp, #48] // 16-byte Folded Spill
; CHECK-NEXT:    stp x30, x9, [sp, #64] // 16-byte Folded Spill
; CHECK-NEXT:    smstop sm
; CHECK-NEXT:    bl callee
; CHECK-NEXT:    smstart sm
; CHECK-NEXT:    ldp d9, d8, [sp, #48] // 16-byte Folded Reload
; CHECK-NEXT:    ldr x30, [sp, #64] // 8-byte Folded Reload
; CHECK-NEXT:    ldp d11, d10, [sp, #32] // 16-byte Folded Reload
; CHECK-NEXT:    ldp d13, d12, [sp, #16] // 16-byte Folded Reload
; CHECK-NEXT:    ldp d15, d14, [sp], #80 // 16-byte Folded Reload
; CHECK-NEXT:    ret
;
; OUTLINER-LABEL: streaming_mode_change3:
; OUTLINER-NOT: OUTLINED_FUNCTION_
;
  call void @callee();
  ret void;
}

attributes #0 = { "aarch64_pstate_sm_enabled" nounwind }
