; RUN: llc < %s -mtriple=thumbv6m-none-eabi -verify-machineinstrs -frame-pointer=none -mattr=+reserve-r6,+reserve-r8 \
; RUN:     -asm-verbose=false | FileCheck --check-prefix=CHECK-INVALID %s

; Reserved low registers should not be used to correct reg deficit.
define <4 x i32> @four_high_four_return_reserved() {
entry:
  ; CHECK-INVALID-NOT: r{{6|8}}
  tail call void asm sideeffect "", "~{r8},~{r9}"()
  %vecinit = insertelement <4 x i32> undef, i32 1, i32 0
  %vecinit11 = insertelement <4 x i32> %vecinit, i32 2, i32 1
  %vecinit12 = insertelement <4 x i32> %vecinit11, i32 3, i32 2
  %vecinit13 = insertelement <4 x i32> %vecinit12, i32 4, i32 3
  ret <4 x i32> %vecinit13
}

