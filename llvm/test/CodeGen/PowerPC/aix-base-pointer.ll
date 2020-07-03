; RUN: llc -mcpu=pwr7 -mattr=-altivec -verify-machineinstrs \
; RUN:   -mtriple=powerpc-unknown-aix < %s  | FileCheck %s --check-prefix 32BIT

; RUN: llc -mcpu=pwr7 -mattr=-altivec -verify-machineinstrs \
; RUN:   -mtriple=powerpc64-unknown-aix < %s | FileCheck %s --check-prefix 64BIT

; Use an overaligned buffer to force base-pointer usage. Test verifies:
; - base pointer register (r30) is saved/defined/restored.
; - stack frame is allocated with correct alignment.
; - Address of %AlignedBuffer is calculated based off offset from the stack
;   pointer.

define void @caller() {
  %AlignedBuffer = alloca [32 x i32], align 32
  %Pointer = getelementptr inbounds [32 x i32], [32 x i32]* %AlignedBuffer, i64 0, i64 0
  call void @callee(i32* %Pointer)
  ret void
}

declare void @callee(i32*)

; 32BIT-LABEL: .caller:
; 32BIT:         stw 30, -8(1)
; 32BIT:         mr 30, 1
; 32BIT:         clrlwi  0, 1, 27
; 32BIT:         subfic 0, 0, -224
; 32BIT:         stwux 1, 1, 0
; 32BIT:         addi 3, 1, 64
; 32BIT:         bl .callee
; 32BIT:         lwz 1, 0(1)
; 32BIT:         lwz 30, -8(1)

; 64BIT-LABEL: .caller:
; 64BIT:         std 30, -16(1)
; 64BIT:         mr 30, 1
; 64BIT:         clrldi  0, 1, 59
; 64BIT:         subfic 0, 0, -288
; 64BIT:         stdux 1, 1, 0
; 64BIT:         addi 3, 1, 128
; 64BIT:         bl .callee
; 64BIT:         ld 1, 0(1)
; 64BIT:         ld 30, -16(1)
