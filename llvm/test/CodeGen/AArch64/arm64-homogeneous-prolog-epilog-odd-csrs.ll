; RUN: llc < %s -mtriple=arm64-apple-ios7.0 -homogeneous-prolog-epilog | FileCheck %s
; RUN: llc < %s -mtriple=aarch64-unknown-linux-gnu  -homogeneous-prolog-epilog | FileCheck %s --check-prefixes=CHECK-LINUX

declare void @bar(i32 %i)

define void @odd_num_callee_saved_registers(ptr swifterror %error, i32 %i) nounwind minsize {
  call void asm sideeffect "mov x0, #42", "~{x0},~{x19},~{x20},~{x22},~{x23},~{x24},~{x25},~{x26},~{x27},~{x28}"() nounwind
  call void @bar(i32 %i)
  ret void
}

define void @odd_num_callee_saved_registers_with_fpr(ptr swifterror %error, i32 %i) nounwind minsize {
  call void asm sideeffect "mov x0, #42", "~{x0},~{x19},~{x20},~{x22},~{x23},~{x24},~{x25},~{x26},~{x27},~{x28},~{d8},~{d9}"() nounwind
  call void @bar(i32 %i)
  ret void
}

; CHECK-LABEL: _OUTLINED_FUNCTION_PROLOG_x30x29x19x20x22x23x24x25x26x27x28:
; CHECK:	str	x28, [sp, #-80]!
; CHECK-LABEL: _OUTLINED_FUNCTION_EPILOG_TAIL_x30x29x19x20x22x23x24x25x26x27x28:
; CHECK:	ldr	x28, [sp], #96

; CHECK-LABEL: _OUTLINED_FUNCTION_PROLOG_x30x29x19x20x22x23x24x25x26x27x28d8d9:
; CHECK:	stp	d9, d8, [sp, #-96]!
; CHECK:	str	x28, [sp, #16]
; CHECK-LABEL: _OUTLINED_FUNCTION_EPILOG_TAIL_x30x29x19x20x22x23x24x25x26x27x28d8d9
; CHECK:	ldr	x28, [sp, #16]
; CHECK:	ldp	d9, d8, [sp], #112

; CHECK-LINUX-NOT: OUTLINED_FUNCTION_PROLOG
; CHECK-LINUX-NOT: OUTLINED_FUNCTION_EPILOG
