; RUN: llc -verify-machineinstrs -enable-machine-outliner -mtriple=aarch64-arm-none-eabi -frame-pointer=non-leaf < %s | FileCheck %s --check-prefix=NOOMIT
; RUN: llc -verify-machineinstrs -enable-machine-outliner -mtriple=aarch64-arm-none-eabi -frame-pointer=none < %s | FileCheck %s --check-prefix=OMITFP

define void @_Z1giii(i32 %x, i32 %y, i32 %z) minsize {
; NOOMIT-LABEL: _Z1giii:
; NOOMIT:       // %bb.0: // %entry
; NOOMIT-NEXT:    b _Z1hiii
;
; OMITFP-LABEL: _Z1giii:
; OMITFP:       // %bb.0: // %entry
; OMITFP-NEXT:    b _Z1hiii
entry:
  tail call void @_Z1hiii(i32 %x, i32 %y, i32 %z)
  ret void
}

declare void @_Z1hiii(i32, i32, i32) minsize

define void @_Z2f1v() minsize {
; NOOMIT-LABEL: _Z2f1v:
; NOOMIT:       // %bb.0: // %entry
; NOOMIT-NEXT:    stp x29, x30, [sp, #-16]! // 16-byte Folded Spill
; NOOMIT-NEXT:    mov x29, sp
; NOOMIT-NEXT:    .cfi_def_cfa w29, 16
; NOOMIT-NEXT:    .cfi_offset w30, -8
; NOOMIT-NEXT:    .cfi_offset w29, -16
; NOOMIT-NEXT:    bl OUTLINED_FUNCTION_0
; NOOMIT-NEXT:    ldp x29, x30, [sp], #16 // 16-byte Folded Reload
; NOOMIT-NEXT:    b _Z1giii
;
; OMITFP-LABEL: _Z2f1v:
; OMITFP:       // %bb.0: // %entry
; OMITFP-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OMITFP-NEXT:    .cfi_def_cfa_offset 16
; OMITFP-NEXT:    .cfi_offset w30, -16
; OMITFP-NEXT:    bl OUTLINED_FUNCTION_0
; OMITFP-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OMITFP-NEXT:    b _Z1giii
entry:
  tail call void @_Z1giii(i32 1, i32 2, i32 3)
  tail call void @_Z1giii(i32 1, i32 2, i32 3)
  ret void
}

define void @_Z2f2v() minsize {
; NOOMIT-LABEL: _Z2f2v:
; NOOMIT:       // %bb.0: // %entry
; NOOMIT-NEXT:    stp x29, x30, [sp, #-16]! // 16-byte Folded Spill
; NOOMIT-NEXT:    mov x29, sp
; NOOMIT-NEXT:    .cfi_def_cfa w29, 16
; NOOMIT-NEXT:    .cfi_offset w30, -8
; NOOMIT-NEXT:    .cfi_offset w29, -16
; NOOMIT-NEXT:    bl OUTLINED_FUNCTION_0
; NOOMIT-NEXT:    ldp x29, x30, [sp], #16 // 16-byte Folded Reload
; NOOMIT-NEXT:    b _Z1giii
;
; OMITFP-LABEL: _Z2f2v:
; OMITFP:       // %bb.0: // %entry
; OMITFP-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OMITFP-NEXT:    .cfi_def_cfa_offset 16
; OMITFP-NEXT:    .cfi_offset w30, -16
; OMITFP-NEXT:    bl OUTLINED_FUNCTION_0
; OMITFP-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OMITFP-NEXT:    b _Z1giii
entry:
  tail call void @_Z1giii(i32 1, i32 2, i32 3)
  tail call void @_Z1giii(i32 1, i32 2, i32 3)
  ret void
}

; OMITFP-LABEL: OUTLINED_FUNCTION_0:
; OMITFP:              .cfi_startproc
; OMITFP-NEXT: // %bb.0:
; OMITFP-NEXT:         .cfi_def_cfa_offset 16
; OMITFP-NEXT:         .cfi_offset w30, -16
; OMITFP-NEXT:         str     x30, [sp, #-16]!
; OMITFP-NEXT:         mov     w0, #1
; OMITFP-NEXT:         mov     w1, #2
; OMITFP-NEXT:         mov     w2, #3
; OMITFP-NEXT:         bl      _Z1giii
; OMITFP-NEXT:         mov     w0, #1
; OMITFP-NEXT:         mov     w1, #2
; OMITFP-NEXT:         mov     w2, #3
; OMITFP-NEXT:         ldr     x30, [sp], #16
; OMITFP-NEXT:         ret

; NOOMIT-LABEL: OUTLINED_FUNCTION_0:
; NOOMIT:              .cfi_startproc
; NOOMIT-NEXT: // %bb.0:
; NOOMIT-NEXT:         .cfi_def_cfa_offset 16
; NOOMIT-NEXT:         .cfi_offset w30, -16
; NOOMIT-NEXT:         str     x30, [sp, #-16]!
; NOOMIT-NEXT:         mov     w0, #1
; NOOMIT-NEXT:         mov     w1, #2
; NOOMIT-NEXT:         mov     w2, #3
; NOOMIT-NEXT:         bl      _Z1giii
; NOOMIT-NEXT:         mov     w0, #1
; NOOMIT-NEXT:         mov     w1, #2
; NOOMIT-NEXT:         mov     w2, #3
; NOOMIT-NEXT:         ldr     x30, [sp], #16
; NOOMIT-NEXT:         ret
