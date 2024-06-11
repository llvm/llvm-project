; RUN: llc -mtriple=aarch64-linux-gnu -mattr=+sve -mattr=+sme2 -enable-machine-outliner -verify-machineinstrs < %s | FileCheck %s

declare void @callee();

define void @streaming_mode_change1() #0 {
; CHECK-LABEL: streaming_mode_change1:
; CHECK:       // %bb.0:
; CHECK-NEXT:    stp d15, d14, [sp, #-80]! // 16-byte Folded Spill
; CHECK-NEXT:    stp d13, d12, [sp, #16] // 16-byte Folded Spill
; CHECK-NEXT:    stp d11, d10, [sp, #32] // 16-byte Folded Spill
; CHECK-NEXT:    stp d9, d8, [sp, #48] // 16-byte Folded Spill
; CHECK-NEXT:    str x30, [sp, #64] // 8-byte Folded Spill
; CHECK-NEXT:    bl OUTLINED_FUNCTION_1
; CHECK-NEXT:    smstart sm
; CHECK-NEXT:    ldp d9, d8, [sp, #48] // 16-byte Folded Reload
; CHECK-NEXT:    ldr x30, [sp, #64] // 8-byte Folded Reload
; CHECK-NEXT:    b OUTLINED_FUNCTION_0
  call void @callee();
  ret void;
}

define void @streaming_mode_change2() #0 {
; CHECK-LABEL: streaming_mode_change2:
; CHECK:       // %bb.0:
; CHECK-NEXT:    stp d15, d14, [sp, #-80]! // 16-byte Folded Spill
; CHECK-NEXT:    stp d13, d12, [sp, #16] // 16-byte Folded Spill
; CHECK-NEXT:    stp d11, d10, [sp, #32] // 16-byte Folded Spill
; CHECK-NEXT:    stp d9, d8, [sp, #48] // 16-byte Folded Spill
; CHECK-NEXT:    str x30, [sp, #64] // 8-byte Folded Spill
; CHECK-NEXT:    bl OUTLINED_FUNCTION_1
; CHECK-NEXT:    smstart sm
; CHECK-NEXT:    ldp d9, d8, [sp, #48] // 16-byte Folded Reload
; CHECK-NEXT:    ldr x30, [sp, #64] // 8-byte Folded Reload
; CHECK-NEXT:    b OUTLINED_FUNCTION_0
  call void @callee();
  ret void;
}

define void @streaming_mode_change3() #0 {
; CHECK-LABEL: streaming_mode_change3:
; CHECK:       // %bb.0:
; CHECK-NEXT:    stp d15, d14, [sp, #-80]! // 16-byte Folded Spill
; CHECK-NEXT:    stp d13, d12, [sp, #16] // 16-byte Folded Spill
; CHECK-NEXT:    stp d11, d10, [sp, #32] // 16-byte Folded Spill
; CHECK-NEXT:    stp d9, d8, [sp, #48] // 16-byte Folded Spill
; CHECK-NEXT:    str x30, [sp, #64] // 8-byte Folded Spill
; CHECK-NEXT:    bl OUTLINED_FUNCTION_1
; CHECK-NEXT:    smstart sm
; CHECK-NEXT:    ldp d9, d8, [sp, #48] // 16-byte Folded Reload
; CHECK-NEXT:    ldr x30, [sp, #64] // 8-byte Folded Reload
; CHECK-NEXT:    b OUTLINED_FUNCTION_0
  call void @callee();
  ret void;
}

; CHECK-LABEL: OUTLINED_FUNCTION_0:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ldp d11, d10, [sp, #32]
; CHECK-NEXT:    ldp d13, d12, [sp, #16]
; CHECK-NEXT:    ldp d15, d14, [sp], #80
; CHECK-NEXT:    ret

; CHECK-LABEL: OUTLINED_FUNCTION_1:
; CHECK:       // %bb.0:
; CHECK-NEXT:    smstop	sm
; CHECK-NEXT:    b callee

attributes #0 = { "aarch64_pstate_sm_enabled" nounwind }
