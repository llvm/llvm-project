; RUN: llc -mcpu=pwr7 -mattr=-altivec -verify-machineinstrs \
; RUN:   -mtriple=powerpc-unknown-aix < %s  | FileCheck %s --check-prefix 32BIT

; RUN: llc -mcpu=pwr7 -mattr=-altivec -verify-machineinstrs \
; RUN:   -mtriple=powerpc64-unknown-aix < %s | FileCheck %s --check-prefix 64BIT

; Use an overaligned buffer to force base-pointer usage. Test verifies:
; - base pointer register (r30) is saved/defined/restored.
; - frame pointer register (r31) is saved/defined/restored.
; - stack frame is allocated with correct alignment.
; - Address of %AlignedBuffer is calculated based off offset from the stack
;   pointer.

define float @caller(float %f) {
  %AlignedBuffer = alloca [32 x i32], align 32
  call void @callee(ptr %AlignedBuffer)
  ret float %f
}

declare void @callee(ptr)

; 32BIT-LABEL: .caller:
; 32BIT:         stw 30, -16(1)
; 32BIT:         mr 30, 1
; 32BIT:         clrlwi  0, 1, 27
; 32BIT:         subfic 0, 0, -224
; 32BIT:         stwux 1, 1, 0
; 32BIT:         addi 3, 1, 64
; 32BIT:         stw 31, -12(30)
; 32BIT:         bl .callee
; 32BIT:         lwz 31, -12(30)
; 32BIT:         mr 1, 30
; 32BIT:         lwz 30, -16(1)

; 64BIT-LABEL: .caller:
; 64BIT:         std 30, -24(1)
; 64BIT:         mr 30, 1
; 64BIT:         clrldi  0, 1, 59
; 64BIT:         subfic 0, 0, -288
; 64BIT:         stdux 1, 1, 0
; 64BIT:         addi 3, 1, 128
; 64BIT:         std 31, -16(30)
; 64BIT:         bl .callee
; 64BIT:         ld 31, -16(30)
; 64BIT:         mr 1, 30
; 64BIT:         ld 30, -24(1)
