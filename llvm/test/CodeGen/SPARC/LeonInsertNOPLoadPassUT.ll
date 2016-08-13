; RUN: llc %s -O0 -march=sparc -mcpu=ut699 -o - | FileCheck %s
; RUN: llc %s -O0 -march=sparc -mcpu=leon3 -mattr=+insertnopload -o - | FileCheck %s

; CHECK-LABEL: ld_float_test
; CHECK:       ld [%o0+%lo(.LCPI0_0)], %f0
; CHECK-NEXT:  nop
define float @ld_float_test() #0 {
entry:
  %f = alloca float, align 4
  store float 0x3FF3C08320000000, float* %f, align 4
  %0 = load float, float* %f, align 4
  ret float %0
}

; CHECK-LABEL: ld_i32_test
; CHECK:       ld [%o0], %o0
; CHECK-NEXT:  nop
define i32 @ld_i32_test(i32 *%p) {
  %res = load i32, i32* %p
  ret i32 %res
}

; CHECK-LABEL: ld_inlineasm_test_1
; CHECK:       ld [%o0], %o0
; CHECK-NEXT:  !NO_APP
; CHECK-NEXT:  nop
define float @ld_inlineasm_test_1(float* %a) {
entry:
  %res = tail call float asm sideeffect "ld [$1], $0", "=r,r"(float* %a)

  ret float %res
}

; CHECK-LABEL: ld_inlineasm_test_2
; CHECK:       ld [%o0], %o0
; CHECK-NEXT:  !NO_APP
; CHECK-NEXT:  nop
define i32 @ld_inlineasm_test_2(i32* %a) {
entry:
  %res = tail call i32 asm sideeffect "ld [$1], $0", "=r,r"(i32* %a)

  ret i32 %res
}