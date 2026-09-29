; RUN: llc -mtriple=thumbv6m-linux-eabi      -mattr=+execute-only %s -o - | FileCheck %s --check-prefixes=CHECK,CHECK-NOMOVW-SIZE,CHECK-NOMOVW-LIMIT
; RUN: llc -mtriple=thumbv8m.base-linux-eabi -mattr=+execute-only %s -o - | FileCheck %s --check-prefixes=CHECK,CHECK-MOVW-SIZE,CHECK-MOVW-LIMIT
; RUN: llc -mtriple=thumbv7m-linux-eabi      -mattr=+execute-only %s -o - | FileCheck %s --check-prefixes=CHECK,CHECK-MOVW-SIZE,CHECK-MRC-LIMIT
; RUN: llc -mtriple=thumbv8m.main-linux-eabi -mattr=+execute-only %s -o - | FileCheck %s --check-prefixes=CHECK,CHECK-MOVW-SIZE,CHECK-MRC-LIMIT

%struct.large_struct = type { [500 x i8] }
declare void @fn(ptr)

; CHECK-LABEL:             test:
; CHECK:                   mov  [[SP:r[0-9]+]], sp
; CHECK-MOVW-SIZE-NEXT:    movw [[SIZE:r[0-9]+]], #4032
; CHECK-NOMOVW-SIZE-NEXT:  movs [[SIZE:r[0-9]+]], #15
; CHECK-NOMOVW-SIZE-NEXT:  lsls [[SIZE]], [[SIZE]], #8
; CHECK-NOMOVW-SIZE-NEXT:  adds [[SIZE]], #192
; CHECK-NEXT:              sub{{s?}} [[SP]], [[SP]], [[SIZE]]
; CHECK-MOVW-LIMIT-NEXT:   movw [[LIMIT:r[0-9]+]], :lower16:__STACK_LIMIT
; CHECK-MOVW-LIMIT-NEXT:   movt [[LIMIT]], :upper16:__STACK_LIMIT
; CHECK-NOMOVW-LIMIT-NEXT: movs [[LIMIT:r[0-9]+]], :upper8_15:__STACK_LIMIT
; CHECK-NOMOVW-LIMIT-NEXT: lsls [[LIMIT]], [[LIMIT]], #8
; CHECK-NOMOVW-LIMIT-NEXT: adds [[LIMIT]], :upper0_7:__STACK_LIMIT
; CHECK-NOMOVW-LIMIT-NEXT: lsls [[LIMIT]], [[LIMIT]], #8
; CHECK-NOMOVW-LIMIT-NEXT: adds [[LIMIT]], :lower8_15:__STACK_LIMIT
; CHECK-NOMOVW-LIMIT-NEXT: lsls [[LIMIT]], [[LIMIT]], #8
; CHECK-NOMOVW-LIMIT-NEXT: adds [[LIMIT]], :lower0_7:__STACK_LIMIT
; CHECK-MRC-LIMIT-NEXT:    p15, #0, [[LIMIT:r[0-9]+]], c13, c0, #3
; CHECK-NEXT:              ldr [[LIMIT]], [[[LIMIT]]{{.*}}]
; CHECK-NEXT:              cmp [[LIMIT]], [[SP]]
; CHECK-NEXT:              bls
; CHECK-NEXT:              @{{.*}}:
; CHECK-MOVW-SIZE-NEXT:    movw r4, #4032
; CHECK-NOMOVW-SIZE-NEXT:  movs r4, #15
; CHECK-NOMOVW-SIZE-NEXT:  lsls r4, r4, #8
; CHECK-NOMOVW-SIZE-NEXT:  adds r4, #192
; CHECK-MOVW-SIZE-NEXT:    movw r5, #484
; CHECK-NOMOVW-SIZE-NEXT:  movs r5, #1
; CHECK-NOMOVW-SIZE-NEXT:  lsls r5, r5, #8
; CHECK-NOMOVW-SIZE-NEXT:  adds r5, #228
; CHECK-NEXT:              push {lr}
; CHECK-NEXT:              bl __morestack

define void @test(ptr byval(%struct.large_struct) align 4 %arg) #0 {
  %ptr = alloca i32, i32 1000
  call void @fn(ptr %ptr)
  ret void
}

attributes #0 = { "split-stack" }
