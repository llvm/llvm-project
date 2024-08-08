; RUN: llc -mtriple=arm64 -global-isel=0 -fast-isel=0         -relocation-model=pic -o - %s -mcpu=cyclone -mattr=+pauth | FileCheck %s
; RUN: llc -mtriple=arm64 -global-isel=0 -fast-isel=1         -relocation-model=pic -o - %s -mcpu=cyclone -mattr=+pauth | FileCheck %s
; RUN: llc -mtriple=arm64 -global-isel=1 -global-isel-abort=1 -relocation-model=pic -o - %s -mcpu=cyclone -mattr=+pauth | FileCheck %s

;; Note: for FastISel, we fall back to SelectionDAG

@var8 = external global i8, align 1

define i8 @test_i8(i8 %new) {
  %val = load i8, ptr @var8, align 1
  store i8 %new, ptr @var8
  ret i8 %val

; CHECK:      adrp x[[HIREG:[0-9]+]], :got_auth:var8
; CHECK-NEXT: add x[[HIREG]], x[[HIREG]], :got_auth_lo12:var8
; CHECK-NEXT: ldr x[[VAR_ADDR:[0-9]+]], [x[[HIREG]]]
; CHECK-NEXT: autda x[[VAR_ADDR]], x[[HIREG]]
; CHECK-NEXT: ldrb {{w[0-9]+}}, [x[[VAR_ADDR]]]
}

!llvm.module.flags = !{!0, !1}
!0 = !{i32 1, !"aarch64-elf-pauthabi-platform", i32 268435458}
!1 = !{i32 1, !"aarch64-elf-pauthabi-version", i32 256}
