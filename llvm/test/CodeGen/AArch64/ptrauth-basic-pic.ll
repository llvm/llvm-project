; RUN: rm -rf %t && split-file %s %t && cd %t
; RUN: cat common.ll authflag.ll > auth.ll
; RUN: cat common.ll authflag-bb.ll > auth-bb.ll

; RUN: llc -mtriple=aarch64-linux-gnu -global-isel=0 -fast-isel=0         -verify-machineinstrs \
; RUN:   -relocation-model=pic -mattr=+pauth -mattr=+fpac auth.ll -o - | FileCheck %s --check-prefixes=CHECK,CHECKA,NOTRAP,NOTRAPA
; RUN: llc -mtriple=aarch64-linux-gnu -global-isel=0 -fast-isel=0         -verify-machineinstrs \
; RUN:   -relocation-model=pic -mattr=+pauth              auth.ll -o - | FileCheck %s --check-prefixes=CHECK,CHECKA,TRAP,TRAPA

; RUN: llc -mtriple=aarch64-linux-gnu -global-isel=0 -fast-isel=1         -verify-machineinstrs \
; RUN:   -relocation-model=pic -mattr=+pauth -mattr=+fpac auth.ll -o - | FileCheck %s --check-prefixes=CHECK,CHECKA,NOTRAP,NOTRAPA
; RUN: llc -mtriple=aarch64-linux-gnu -global-isel=0 -fast-isel=1         -verify-machineinstrs \
; RUN:   -relocation-model=pic -mattr=+pauth              auth.ll -o - | FileCheck %s --check-prefixes=CHECK,CHECKA,TRAP,TRAPA

; RUN: llc -mtriple=aarch64-linux-gnu -global-isel=1 -global-isel-abort=1 -verify-machineinstrs \
; RUN:   -relocation-model=pic -mattr=+pauth -mattr=+fpac auth.ll -o - | FileCheck %s --check-prefixes=CHECK,CHECKA,NOTRAP,NOTRAPA
; RUN: llc -mtriple=aarch64-linux-gnu -global-isel=1 -global-isel-abort=1 -verify-machineinstrs \
; RUN:   -relocation-model=pic -mattr=+pauth              auth.ll -o - | FileCheck %s --check-prefixes=CHECK,CHECKA,TRAP,TRAPA

; RUN: llc -mtriple=aarch64-linux-gnu -global-isel=0 -fast-isel=0         -verify-machineinstrs \
; RUN:   -relocation-model=pic -mattr=+pauth -mattr=+fpac auth-bb.ll -o - | FileCheck %s --check-prefixes=CHECK,CHECKB,NOTRAP,NOTRAPB
; RUN: llc -mtriple=aarch64-linux-gnu -global-isel=0 -fast-isel=0         -verify-machineinstrs \
; RUN:   -relocation-model=pic -mattr=+pauth              auth-bb.ll -o - | FileCheck %s --check-prefixes=CHECK,CHECKB,TRAP,TRAPB

; RUN: llc -mtriple=aarch64-linux-gnu -global-isel=0 -fast-isel=1         -verify-machineinstrs \
; RUN:   -relocation-model=pic -mattr=+pauth -mattr=+fpac auth-bb.ll -o - | FileCheck %s --check-prefixes=CHECK,CHECKB,NOTRAP,NOTRAPB
; RUN: llc -mtriple=aarch64-linux-gnu -global-isel=0 -fast-isel=1         -verify-machineinstrs \
; RUN:   -relocation-model=pic -mattr=+pauth              auth-bb.ll -o - | FileCheck %s --check-prefixes=CHECK,CHECKB,TRAP,TRAPB

; RUN: llc -mtriple=aarch64-linux-gnu -global-isel=1 -global-isel-abort=1 -verify-machineinstrs \
; RUN:   -relocation-model=pic -mattr=+pauth -mattr=+fpac auth-bb.ll -o - | FileCheck %s --check-prefixes=CHECK,CHECKB,NOTRAP,NOTRAPB
; RUN: llc -mtriple=aarch64-linux-gnu -global-isel=1 -global-isel-abort=1 -verify-machineinstrs \
; RUN:   -relocation-model=pic -mattr=+pauth              auth-bb.ll -o - | FileCheck %s --check-prefixes=CHECK,CHECKB,TRAP,TRAPB

;; Note: for FastISel, we fall back to SelectionDAG

;--- common.ll
@var = global i32 0

define i32 @get_globalvar() {
; CHECK-LABEL: get_globalvar:
; CHECK:         adrp  x17, :got_auth:var
; CHECK-NEXT:    add   x17, x17, :got_auth_lo12:var
; NOTRAP-NEXT:   ldr   x8,  [x17]
; NOTRAPA-NEXT:  autda x8,  x17
; NOTRAPB-NEXT:  autdb x8,  x17
; TRAP-NEXT:     ldr   x16, [x17]
; TRAPA-NEXT:    autda x16, x17
; TRAPB-NEXT:    autdb x16, x17
; TRAP-NEXT:     mov   x17, x16
; TRAP-NEXT:     xpacd x17
; TRAP-NEXT:     cmp   x16, x17
; TRAP-NEXT:     b.eq  .Lauth_success_0
; TRAPA-NEXT:    brk   #0xc472
; TRAPB-NEXT:    brk   #0xc473
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
; NOTRAPA-NEXT:  autda x0,  x17
; NOTRAPB-NEXT:  autdb x0,  x17
; TRAP-NEXT:     ldr   x16, [x17]
; TRAPA-NEXT:    autda x16, x17
; TRAPB-NEXT:    autdb x16, x17
; TRAP-NEXT:     mov   x17, x16
; TRAP-NEXT:     xpacd x17
; TRAP-NEXT:     cmp   x16, x17
; TRAP-NEXT:     b.eq  .Lauth_success_1
; TRAPA-NEXT:    brk   #0xc472
; TRAPB-NEXT:    brk   #0xc473
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
; CHECKA-NEXT:   autia x16, x17
; CHECKB-NEXT:   autib x16, x17
; TRAP-NEXT:     mov   x17, x16
; TRAP-NEXT:     xpaci x17
; TRAP-NEXT:     cmp   x16, x17
; TRAP-NEXT:     b.eq  .Lauth_success_2
; TRAPA-NEXT:    brk   #0xc470
; TRAPB-NEXT:    brk   #0xc471
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
; CHECKA-NEXT:   autda x16, x17
; CHECKB-NEXT:   autdb x16, x17
; TRAP-NEXT:     mov   x17, x16
; TRAP-NEXT:     xpacd x17
; TRAP-NEXT:     cmp   x16, x17
; TRAP-NEXT:     b.eq  .Lauth_success_3
; TRAPA-NEXT:    brk   #0xc472
; TRAPB-NEXT:    brk   #0xc473
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
; CHECKA-NEXT:   autda x16, x17
; CHECKB-NEXT:   autdb x16, x17
; TRAP-NEXT:     mov   x17, x16
; TRAP-NEXT:     xpacd x17
; TRAP-NEXT:     cmp   x16, x17
; TRAP-NEXT:     b.eq  .Lauth_success_4
; TRAPA-NEXT:    brk   #0xc472
; TRAPB-NEXT:    brk   #0xc473
; TRAP-NEXT:   .Lauth_success_4:
; CHECK-NEXT:    add   x16, x16, #16
; CHECK-NEXT:    mov   x17, #44
; CHECK-NEXT:   pacda x16, x17
; CHECK-NEXT:    mov   x0,  x16
; CHECK-NEXT:    ret

  ret ptr ptrauth (ptr getelementptr (i8, ptr @var, i64 16), i32 2, i64 44)
}

;--- authflag.ll
!llvm.module.flags = !{!0}
!0 = !{i32 8, !"ptrauth-elf-got", i32 1}

;--- authflag-bb.ll
!llvm.module.flags = !{!0, !1}
!0 = !{i32 8, !"ptrauth-elf-got", i32 1}
!1 = !{i32 1, !"ptrauth-keys", !"ibdbibdb"}
