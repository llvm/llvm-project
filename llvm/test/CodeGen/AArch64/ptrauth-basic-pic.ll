; RUN: llc -mtriple=aarch64-linux-gnu -global-isel=0 -fast-isel=0         -verify-machineinstrs \
; RUN:   -relocation-model=pic -mattr=+pauth -mattr=+fpac %s -o - | FileCheck %s --check-prefixes=CHECK,NOTRAP
; RUN: llc -mtriple=aarch64-linux-gnu -global-isel=0 -fast-isel=0         -verify-machineinstrs \
; RUN:   -relocation-model=pic -mattr=+pauth              %s -o - | FileCheck %s --check-prefixes=CHECK,TRAP

; RUN: llc -mtriple=aarch64-linux-gnu -global-isel=0 -fast-isel=1         -verify-machineinstrs \
; RUN:   -relocation-model=pic -mattr=+pauth -mattr=+fpac %s -o - | FileCheck %s --check-prefixes=CHECK,NOTRAP
; RUN: llc -mtriple=aarch64-linux-gnu -global-isel=0 -fast-isel=1         -verify-machineinstrs \
; RUN:   -relocation-model=pic -mattr=+pauth              %s -o - | FileCheck %s --check-prefixes=CHECK,TRAP

; RUN: llc -mtriple=aarch64-linux-gnu -global-isel=1 -global-isel-abort=1 -verify-machineinstrs \
; RUN:   -relocation-model=pic -mattr=+pauth -mattr=+fpac %s -o - | FileCheck %s --check-prefixes=CHECK,NOTRAP
; RUN: llc -mtriple=aarch64-linux-gnu -global-isel=1 -global-isel-abort=1 -verify-machineinstrs \
; RUN:   -relocation-model=pic -mattr=+pauth              %s -o - | FileCheck %s --check-prefixes=CHECK,TRAP

;; Note: for FastISel, we fall back to SelectionDAG

@var = global i32 0

define i32 @get_globalvar() {
; CHECK-LABEL: get_globalvar:
; CHECK:         adrp  x17, :got_auth:var
; CHECK-NEXT:    add   x17, x17, :got_auth_lo12:var
; NOTRAP-NEXT:   ldr   x8,  [x17]
; NOTRAP-NEXT:   autda x8,  x17
; TRAP-NEXT:     ldr   x16, [x17]
; TRAP-NEXT:     autda x16, x17
; TRAP-NEXT:     mov   x17, x16
; TRAP-NEXT:     xpacd x17
; TRAP-NEXT:     cmp   x16, x17
; TRAP-NEXT:     b.eq  .Lauth_success_0
; TRAP-NEXT:     brk   #0xc472
; TRAP-NEXT:   .Lauth_success_0:
; TRAP-NEXT:     mov   x8,  x16
; CHECK-NEXT:    ldr   w0,  [x8]
; CHECK-NEXT:    ret

  %val = load i32, ptr @var
  ret i32 %val
}

define ptr @get_globalvaraddr() {
; CHECK-LABEL: get_globalvaraddr:
; CHECK:         adrp  x17, :got_auth:var
; CHECK-NEXT:    add   x17, x17, :got_auth_lo12:var
; NOTRAP-NEXT:   ldr   x0,  [x17]
; NOTRAP-NEXT:   autda x0,  x17
; TRAP-NEXT:     ldr   x16, [x17]
; TRAP-NEXT:     autda x16, x17
; TRAP-NEXT:     mov   x17, x16
; TRAP-NEXT:     xpacd x17
; TRAP-NEXT:     cmp   x16, x17
; TRAP-NEXT:     b.eq  .Lauth_success_1
; TRAP-NEXT:     brk   #0xc472
; TRAP-NEXT:   .Lauth_success_1:
; TRAP-NEXT:     mov   x0,  x16
; CHECK-NEXT:    ret

  %val = load i32, ptr @var
  ret ptr @var
}

declare i32 @foo()

define ptr @resign_globalfunc() {
; CHECK-LABEL: resign_globalfunc:
; CHECK:         adrp  x17, :got_auth:foo
; CHECK-NEXT:    add   x17, x17, :got_auth_lo12:foo
; CHECK-NEXT:    ldr   x16, [x17]
; CHECK-NEXT:    autia x16, x17
; TRAP-NEXT:     mov   x17, x16
; TRAP-NEXT:     xpaci x17
; TRAP-NEXT:     cmp   x16, x17
; TRAP-NEXT:     b.eq  .Lauth_success_2
; TRAP-NEXT:     brk   #0xc470
; TRAP-NEXT:   .Lauth_success_2:
; CHECK-NEXT:    mov   x17, #42
; CHECK-NEXT:    pacia x16, x17
; CHECK-NEXT:    mov   x0,  x16
; CHECK-NEXT:    ret

  ret ptr ptrauth (ptr @foo, i32 0, i64 42)
}

define ptr @resign_globalvar() {
; CHECK-LABEL: resign_globalvar:
; CHECK:         adrp  x17, :got_auth:var
; CHECK-NEXT:    add   x17, x17, :got_auth_lo12:var
; CHECK-NEXT:    ldr   x16, [x17]
; CHECK-NEXT:    autda x16, x17
; TRAP-NEXT:     mov   x17, x16
; TRAP-NEXT:     xpacd x17
; TRAP-NEXT:     cmp   x16, x17
; TRAP-NEXT:     b.eq  .Lauth_success_3
; TRAP-NEXT:     brk   #0xc472
; TRAP-NEXT:   .Lauth_success_3:
; CHECK-NEXT:    mov   x17, #43
; CHECK-NEXT:    pacdb x16, x17
; CHECK-NEXT:    mov   x0,  x16
; CHECK-NEXT:    ret

  ret ptr ptrauth (ptr @var, i32 3, i64 43)
}

define ptr @resign_globalvar_offset() {
; CHECK-LABEL: resign_globalvar_offset:
; CHECK:         adrp  x17, :got_auth:var
; CHECK-NEXT:    add   x17, x17, :got_auth_lo12:var
; CHECK-NEXT:    ldr   x16, [x17]
; CHECK-NEXT:    autda x16, x17
; TRAP-NEXT:     mov   x17, x16
; TRAP-NEXT:     xpacd x17
; TRAP-NEXT:     cmp   x16, x17
; TRAP-NEXT:     b.eq  .Lauth_success_4
; TRAP-NEXT:     brk   #0xc472
; TRAP-NEXT:   .Lauth_success_4:
; CHECK-NEXT:    add   x16, x16, #16
; CHECK-NEXT:    mov   x17, #44
; CHECK-NEXT:    pacda x16, x17
; CHECK-NEXT:    mov   x0,  x16
; CHECK-NEXT:    ret

  ret ptr ptrauth (ptr getelementptr (i8, ptr @var, i64 16), i32 2, i64 44)
}

!llvm.module.flags = !{!0}
!0 = !{i32 8, !"ptrauth-elf-got", i32 1}
