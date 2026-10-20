; RUN: rm -rf %t && split-file %s %t

;--- err1.ll

; RUN: not --crash llc %t/err1.ll -mtriple=aarch64-elf \
; RUN:   -aarch64-min-jump-table-entries=1 -aarch64-enable-atomic-cfg-tidy=0 \
; RUN:   -code-model=large \
; RUN:   -o - -verify-machineinstrs 2>&1 | FileCheck %s --check-prefix=ERR1

; RUN: not --crash llc %t/err1.ll -mtriple=aarch64-elf \
; RUN:   -aarch64-min-jump-table-entries=1 -aarch64-enable-atomic-cfg-tidy=0 \
; RUN:   -global-isel -global-isel-abort=1 \
; RUN:   -code-model=large \
; RUN:   -o - -verify-machineinstrs 2>&1 | FileCheck %s --check-prefix=ERR1

; ERR1: LLVM ERROR: Unsupported code-model for hardened jump-table
define i32 @test_jumptable(i32 %in) "aarch64-jump-table-hardening" {

  switch i32 %in, label %def [
    i32 0, label %lbl1
    i32 1, label %lbl2
  ]

def:
  ret i32 0

lbl1:
  ret i32 1

lbl2:
  ret i32 2
}

;--- test.ll

; RUN: llc %t/test.ll -mtriple=arm64-apple-darwin -aarch64-enable-collect-loh=0 \
; RUN:   -aarch64-min-jump-table-entries=1 -aarch64-enable-atomic-cfg-tidy=0 \
; RUN:   -o - -verify-machineinstrs | FileCheck %s --check-prefix=MACHO

; RUN: llc %t/test.ll -mtriple=arm64-apple-darwin -aarch64-enable-collect-loh=0 \
; RUN:   -aarch64-min-jump-table-entries=1 -aarch64-enable-atomic-cfg-tidy=0 \
; RUN:   -global-isel -global-isel-abort=1 \
; RUN:   -o - -verify-machineinstrs | FileCheck %s --check-prefix=MACHO

; RUN: llc %t/test.ll -mtriple=arm64-apple-darwin -aarch64-enable-collect-loh=0 \
; RUN:   -aarch64-min-jump-table-entries=1 -aarch64-enable-atomic-cfg-tidy=0 \
; RUN:   -code-model=large \
; RUN:   -o - -verify-machineinstrs | FileCheck %s --check-prefix=MACHO

; RUN: llc %t/test.ll -mtriple=arm64-apple-darwin -aarch64-enable-collect-loh=0 \
; RUN:   -aarch64-min-jump-table-entries=1 -aarch64-enable-atomic-cfg-tidy=0 \
; RUN:   -global-isel -global-isel-abort=1 \
; RUN:   -code-model=large \
; RUN:   -o - -verify-machineinstrs | FileCheck %s --check-prefix=MACHO

; RUN: llc %t/test.ll -mtriple=aarch64-elf \
; RUN:   -aarch64-min-jump-table-entries=1 -aarch64-enable-atomic-cfg-tidy=0 \
; RUN:   -o - -verify-machineinstrs | FileCheck %s --check-prefix=ELF

; RUN: llc %t/test.ll -mtriple=aarch64-elf \
; RUN:   -aarch64-min-jump-table-entries=1 -aarch64-enable-atomic-cfg-tidy=0 \
; RUN:   -global-isel -global-isel-abort=1 \
; RUN:   -o - -verify-machineinstrs | FileCheck %s --check-prefix=ELF

; MACHO-LABEL: test_jumptable:
; MACHO:        mov   w16, w0
; MACHO:        cmp   x16, #5
; MACHO:        csel  x16, x16, xzr, ls
; MACHO-NEXT:   adrp  x17, LJTI0_0@PAGE
; MACHO-NEXT:   add   x17, x17, LJTI0_0@PAGEOFF
; MACHO-NEXT:   ldrsw x16, [x17, x16, lsl #2]
; MACHO-NEXT:  Ltmp0:
; MACHO-NEXT:   adr   x17, Ltmp0
; MACHO-NEXT:   add   x16, x17, x16
; MACHO-NEXT:   br    x16

; ELF-LABEL: test_jumptable:
; ELF:        mov   w16, w0
; ELF:        cmp   x16, #5
; ELF:        csel  x16, x16, xzr, ls
; ELF-NEXT:   adrp  x17, .LJTI0_0
; ELF-NEXT:   add   x17, x17, :lo12:.LJTI0_0
; ELF-NEXT:   ldrsw x16, [x17, x16, lsl #2]
; ELF-NEXT:  .Ltmp0:
; ELF-NEXT:   adr   x17, .Ltmp0
; ELF-NEXT:   add   x16, x17, x16
; ELF-NEXT:   br    x16

define i32 @test_jumptable(i32 %in) "aarch64-jump-table-hardening" {

  switch i32 %in, label %def [
    i32 0, label %lbl1
    i32 1, label %lbl2
    i32 2, label %lbl3
    i32 4, label %lbl4
    i32 5, label %lbl5
  ]

def:
  ret i32 0

lbl1:
  ret i32 1

lbl2:
  ret i32 2

lbl3:
  ret i32 4

lbl4:
  ret i32 8

lbl5:
  ret i32 10

}

; MACHO-LABEL: LJTI0_0:
; MACHO-NEXT: .long LBB{{[0-9_]+}}-Ltmp0
; MACHO-NEXT: .long LBB{{[0-9_]+}}-Ltmp0
; MACHO-NEXT: .long LBB{{[0-9_]+}}-Ltmp0
; MACHO-NEXT: .long LBB{{[0-9_]+}}-Ltmp0
; MACHO-NEXT: .long LBB{{[0-9_]+}}-Ltmp0
; MACHO-NEXT: .long LBB{{[0-9_]+}}-Ltmp0

; ELF-LABEL: .LJTI0_0:
; ELF-NEXT: .word .LBB{{[0-9_]+}}-.Ltmp0
; ELF-NEXT: .word .LBB{{[0-9_]+}}-.Ltmp0
; ELF-NEXT: .word .LBB{{[0-9_]+}}-.Ltmp0
; ELF-NEXT: .word .LBB{{[0-9_]+}}-.Ltmp0
; ELF-NEXT: .word .LBB{{[0-9_]+}}-.Ltmp0
; ELF-NEXT: .word .LBB{{[0-9_]+}}-.Ltmp0
