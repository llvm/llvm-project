; RUN: llc -mtriple=arm64 -global-isel=0 -fast-isel=0         -relocation-model=pic -o - %s \
; RUN:   -mcpu=cyclone -mattr=+pauth | FileCheck %s
; RUN: llc -mtriple=arm64 -global-isel=0 -fast-isel=1         -relocation-model=pic -o - %s \
; RUN:   -mcpu=cyclone -mattr=+pauth | FileCheck %s
; RUN: llc -mtriple=arm64 -global-isel=1 -global-isel-abort=1 -relocation-model=pic -o - %s \
; RUN:   -mcpu=cyclone -mattr=+pauth | FileCheck %s

;; Note: for FastISel, we fall back to SelectionDAG

@var8 = external global i8, align 1

define i8 @test_i8(i8 %new) {
  %val = load i8, ptr @var8, align 1
  store i8 %new, ptr @var8
  ret i8 %val

; CHECK-LABEL: test_i8:
; CHECK:         adrp  x16, :got_auth:var8
; CHECK-NEXT:    add   x16, x16, :got_auth_lo12:var8
; CHECK-NEXT:    ldr   x9,  [x16]
; CHECK-NEXT:    autda x9,  x16
; CHECK-NEXT:    ldrb  w8,  [x9]
; CHECK-NEXT:    strb  w0,  [x9]
; CHECK-NEXT:    mov   x0,  x8
; CHECK-NEXT:    ret
}

!llvm.module.flags = !{!0}
!0 = !{i32 8, !"ptrauth-elf-got", i32 1}
