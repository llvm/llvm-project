; RUN: llc < %s -mtriple arm64e-apple-darwin -global-isel=0                    -verify-machineinstrs \
; RUN:   -aarch64-ptrauth-auth-checks=none | FileCheck %s -DL="L" --check-prefixes=UNCHECKED,UNCHECKED-DARWIN
; RUN: llc < %s -mtriple arm64e-apple-darwin -global-isel -global-isel-abort=1 -verify-machineinstrs \
; RUN:   -aarch64-ptrauth-auth-checks=none | FileCheck %s -DL="L" --check-prefixes=UNCHECKED,UNCHECKED-DARWIN

; RUN: llc < %s -mtriple arm64e-apple-darwin -global-isel=0                    -verify-machineinstrs \
; RUN:                                     | FileCheck %s -DL="L" --check-prefixes=CHECKED,CHECKED-DARWIN
; RUN: llc < %s -mtriple arm64e-apple-darwin -global-isel -global-isel-abort=1 -verify-machineinstrs \
; RUN:                                     | FileCheck %s -DL="L" --check-prefixes=CHECKED,CHECKED-DARWIN

; RUN: llc < %s -mtriple arm64e-apple-darwin -global-isel=0                    -verify-machineinstrs \
; RUN:   -aarch64-ptrauth-auth-checks=trap | FileCheck %s -DL="L" --check-prefixes=TRAP,TRAP-DARWIN
; RUN: llc < %s -mtriple arm64e-apple-darwin -global-isel -global-isel-abort=1 -verify-machineinstrs \
; RUN:   -aarch64-ptrauth-auth-checks=trap | FileCheck %s -DL="L" --check-prefixes=TRAP,TRAP-DARWIN

; RUN: llc < %s -mtriple aarch64-linux-gnu -mattr=+pauth -global-isel=0                    -verify-machineinstrs \
; RUN:   -aarch64-ptrauth-auth-checks=none | FileCheck %s -DL=".L" --check-prefixes=UNCHECKED,UNCHECKED-ELF
; RUN: llc < %s -mtriple aarch64-linux-gnu -mattr=+pauth -global-isel -global-isel-abort=1 -verify-machineinstrs \
; RUN:   -aarch64-ptrauth-auth-checks=none | FileCheck %s -DL=".L" --check-prefixes=UNCHECKED,UNCHECKED-ELF

; RUN: llc < %s -mtriple aarch64-linux-gnu -mattr=+pauth -global-isel=0                    -verify-machineinstrs \
; RUN:                                     | FileCheck %s -DL=".L" --check-prefixes=CHECKED,CHECKED-ELF
; RUN: llc < %s -mtriple aarch64-linux-gnu -mattr=+pauth -global-isel -global-isel-abort=1 -verify-machineinstrs \
; RUN:                                     | FileCheck %s -DL=".L" --check-prefixes=CHECKED,CHECKED-ELF

; RUN: llc < %s -mtriple aarch64-linux-gnu -mattr=+pauth -global-isel=0                    -verify-machineinstrs \
; RUN:   -aarch64-ptrauth-auth-checks=trap | FileCheck %s -DL=".L" --check-prefixes=TRAP,TRAP-ELF
; RUN: llc < %s -mtriple aarch64-linux-gnu -mattr=+pauth -global-isel -global-isel-abort=1 -verify-machineinstrs \
; RUN:   -aarch64-ptrauth-auth-checks=trap | FileCheck %s -DL=".L" --check-prefixes=TRAP,TRAP-ELF

; Make sure codegen at -O0 does not crash:
;
; RUN: llc < %s -mtriple aarch64-linux-gnu -mattr=+pauth -O0 -verify-machineinstrs -global-isel=0
; RUN: llc < %s -mtriple aarch64-linux-gnu -mattr=+pauth -O0 -verify-machineinstrs -global-isel=1 -global-isel-abort=1

target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"

define i64 @test_auth_blend(i64 %arg, i64 %arg1) {
; UNCHECKED-LABEL: test_auth_blend:
; UNCHECKED:          %bb.0:
; UNCHECKED-DARWIN-NEXT: mov x16, x0
; UNCHECKED-DARWIN-NEXT: mov x17, x1
; UNCHECKED-DARWIN-NEXT: movk x17, #65535, lsl #48
; UNCHECKED-DARWIN-NEXT: autda x16, x17
; UNCHECKED-DARWIN-NEXT: mov x0, x16
; UNCHECKED-ELF-NEXT:    movk x1, #65535, lsl #48
; UNCHECKED-ELF-NEXT:    autda x0, x1
; UNCHECKED-NEXT:        ret
;
; CHECKED-LABEL: test_auth_blend:
; CHECKED:           %bb.0:
; CHECKED-DARWIN-NEXT: mov x16, x0
; CHECKED-DARWIN-NEXT: mov x17, x1
; CHECKED-DARWIN-NEXT: movk x17, #65535, lsl #48
; CHECKED-DARWIN-NEXT: autda x16, x17
; CHECKED-DARWIN-NEXT: mov x0, x16
; CHECKED-ELF-NEXT:    movk x1, #65535, lsl #48
; CHECKED-ELF-NEXT:    autda x0, x1
; CHECKED-NEXT:    ret
;
; TRAP-LABEL: test_auth_blend:
; TRAP:       %bb.0:
; TRAP-DARWIN-NEXT: mov x16, x0
; TRAP-DARWIN-NEXT: mov x17, x1
; TRAP-DARWIN-NEXT: movk x17, #65535, lsl #48
; TRAP-DARWIN-NEXT: autda x16, x17
; TRAP-DARWIN-NEXT: mov x17, x16
; TRAP-DARWIN-NEXT: xpacd x17
; TRAP-DARWIN-NEXT: cmp x16, x17
; TRAP-ELF-NEXT: movk x1, #65535, lsl #48
; TRAP-ELF-NEXT: autda x0, x1
; TRAP-ELF-NEXT: mov x8, x0
; TRAP-ELF-NEXT: xpacd x8
; TRAP-ELF-NEXT: cmp x0, x8
; TRAP-NEXT:    b.eq [[L]]auth_success_0
; TRAP-NEXT:    brk #0xc472
; TRAP-NEXT:  Lauth_success_0:
; TRAP-DARWIN-NEXT:    mov x0, x16
; TRAP-NEXT:    ret
  %tmp0 = call i64 @llvm.ptrauth.blend(i64 %arg1, i64 65535)
  %tmp1 = call i64 @llvm.ptrauth.auth(i64 %arg, i32 2, i64 %tmp0)
  ret i64 %tmp1
}

define i64 @test_resign_blend(i64 %arg, i64 %arg1, i64 %arg2) {
; UNCHECKED-LABEL: test_resign_blend:
; UNCHECKED:       %bb.0:
; UNCHECKED-NEXT:    mov x16, x0
; UNCHECKED-ELF-NEXT:    movk x1, #12345, lsl #48
; UNCHECKED-ELF-NEXT:    autda x16, x1
; UNCHECKED-ELF-NEXT:    movk x2, #56789, lsl #48
; UNCHECKED-ELF-NEXT:    pacdb x16, x2
; UNCHECKED-DARWIN-NEXT: mov x17, x1
; UNCHECKED-DARWIN-NEXT: movk x17, #12345, lsl #48
; UNCHECKED-DARWIN-NEXT: autda x16, x17
; UNCHECKED-DARWIN-NEXT: mov x17, x2
; UNCHECKED-DARWIN-NEXT: movk x17, #56789, lsl #48
; UNCHECKED-DARWIN-NEXT: pacdb x16, x17
; UNCHECKED-NEXT:    mov x0, x16
; UNCHECKED-NEXT:    ret
;
; CHECKED-LABEL: test_resign_blend:
; CHECKED:       %bb.0:
; CHECKED-NEXT:    mov x16, x0
; CHECKED-ELF-NEXT:    movk x1, #12345, lsl #48
; CHECKED-ELF-NEXT:    autda x16, x1
; CHECKED-DARWIN-NEXT: mov x17, x1
; CHECKED-DARWIN-NEXT: movk x17, #12345, lsl #48
; CHECKED-DARWIN-NEXT: autda x16, x17
; CHECKED-NEXT:    mov x17, x16
; CHECKED-NEXT:    xpacd x17
; CHECKED-NEXT:    cmp x16, x17
; CHECKED-NEXT:    b.eq [[L]]auth_success_0
; CHECKED-NEXT:    mov x16, x17
; CHECKED-NEXT:    b [[L]]resign_end_0
; CHECKED-NEXT:  Lauth_success_0:
; CHECKED-ELF-NEXT:    movk x2, #56789, lsl #48
; CHECKED-ELF-NEXT:    pacdb x16, x2
; CHECKED-DARWIN-NEXT: mov x17, x2
; CHECKED-DARWIN-NEXT: movk x17, #56789, lsl #48
; CHECKED-DARWIN-NEXT: pacdb x16, x17
; CHECKED-NEXT:  Lresign_end_0:
; CHECKED-NEXT:    mov x0, x16
; CHECKED-NEXT:    ret
;
; TRAP-LABEL: test_resign_blend:
; TRAP:       %bb.0:
; TRAP-NEXT:    mov x16, x0
; TRAP-ELF-NEXT:    movk x1, #12345, lsl #48
; TRAP-ELF-NEXT:    autda x16, x1
; TRAP-DARWIN-NEXT: mov x17, x1
; TRAP-DARWIN-NEXT: movk x17, #12345, lsl #48
; TRAP-DARWIN-NEXT: autda x16, x17
; TRAP-NEXT:    mov x17, x16
; TRAP-NEXT:    xpacd x17
; TRAP-NEXT:    cmp x16, x17
; TRAP-NEXT:    b.eq [[L]]auth_success_1
; TRAP-NEXT:    brk #0xc472
; TRAP-NEXT:  Lauth_success_1:
; TRAP-ELF-NEXT:    movk x2, #56789, lsl #48
; TRAP-ELF-NEXT:    pacdb x16, x2
; TRAP-DARWIN-NEXT: mov x17, x2
; TRAP-DARWIN-NEXT: movk x17, #56789, lsl #48
; TRAP-DARWIN-NEXT: pacdb x16, x17
; TRAP-NEXT:    mov x0, x16
; TRAP-NEXT:    ret
  %tmp0 = call i64 @llvm.ptrauth.blend(i64 %arg1, i64 12345)
  %tmp1 = call i64 @llvm.ptrauth.blend(i64 %arg2, i64 56789)
  %tmp2 = call i64 @llvm.ptrauth.resign(i64 %arg, i32 2, i64 %tmp0, i32 3, i64 %tmp1)
  ret i64 %tmp2
}

define i64 @test_resign_blend_and_const(i64 %arg, i64 %arg1) {
; UNCHECKED-LABEL: test_resign_blend_and_const:
; UNCHECKED:       %bb.0:
; UNCHECKED-NEXT:    mov x16, x0
; UNCHECKED-ELF-NEXT:    movk x1, #12345, lsl #48
; UNCHECKED-ELF-NEXT:    autda x16, x1
; UNCHECKED-DARWIN-NEXT: mov x17, x1
; UNCHECKED-DARWIN-NEXT: movk x17, #12345, lsl #48
; UNCHECKED-DARWIN-NEXT: autda x16, x17
; UNCHECKED-NEXT:    mov x17, #56789
; UNCHECKED-NEXT:    pacdb x16, x17
; UNCHECKED-NEXT:    mov x0, x16
; UNCHECKED-NEXT:    ret
;
; CHECKED-LABEL: test_resign_blend_and_const:
; CHECKED:       %bb.0:
; CHECKED-NEXT:    mov x16, x0
; CHECKED-ELF-NEXT:    movk x1, #12345, lsl #48
; CHECKED-ELF-NEXT:    autda x16, x1
; CHECKED-DARWIN-NEXT: mov x17, x1
; CHECKED-DARWIN-NEXT: movk x17, #12345, lsl #48
; CHECKED-DARWIN-NEXT: autda x16, x17
; CHECKED-NEXT:    mov x17, x16
; CHECKED-NEXT:    xpacd x17
; CHECKED-NEXT:    cmp x16, x17
; CHECKED-NEXT:    b.eq [[L]]auth_success_1
; CHECKED-NEXT:    mov x16, x17
; CHECKED-NEXT:    b [[L]]resign_end_1
; CHECKED-NEXT:  Lauth_success_1:
; CHECKED-NEXT:    mov x17, #56789
; CHECKED-NEXT:    pacdb x16, x17
; CHECKED-NEXT:  Lresign_end_1:
; CHECKED-NEXT:    mov x0, x16
; CHECKED-NEXT:    ret
;
; TRAP-LABEL: test_resign_blend_and_const:
; TRAP:       %bb.0:
; TRAP-NEXT:    mov x16, x0
; TRAP-ELF-NEXT:    movk x1, #12345, lsl #48
; TRAP-ELF-NEXT:    autda x16, x1
; TRAP-DARWIN-NEXT: mov x17, x1
; TRAP-DARWIN-NEXT: movk x17, #12345, lsl #48
; TRAP-DARWIN-NEXT: autda x16, x17
; TRAP-NEXT:    mov x17, x16
; TRAP-NEXT:    xpacd x17
; TRAP-NEXT:    cmp x16, x17
; TRAP-NEXT:    b.eq [[L]]auth_success_2
; TRAP-NEXT:    brk #0xc472
; TRAP-NEXT:  Lauth_success_2:
; TRAP-NEXT:    mov x17, #56789
; TRAP-NEXT:    pacdb x16, x17
; TRAP-NEXT:    mov x0, x16
; TRAP-NEXT:    ret
  %tmp0 = call i64 @llvm.ptrauth.blend(i64 %arg1, i64 12345)
  %tmp1 = call i64 @llvm.ptrauth.resign(i64 %arg, i32 2, i64 %tmp0, i32 3, i64 56789)
  ret i64 %tmp1
}

define i64 @test_resign_blend_and_addr(i64 %arg, i64 %arg1, i64 %arg2) {
; UNCHECKED-LABEL: test_resign_blend_and_addr:
; UNCHECKED:       %bb.0:
; UNCHECKED-NEXT:    mov x16, x0
; UNCHECKED-ELF-NEXT:    movk x1, #12345, lsl #48
; UNCHECKED-ELF-NEXT:    autda x16, x1
; UNCHECKED-DARWIN-NEXT: mov x17, x1
; UNCHECKED-DARWIN-NEXT: movk x17, #12345, lsl #48
; UNCHECKED-DARWIN-NEXT: autda x16, x17
; UNCHECKED-NEXT:    pacdb x16, x2
; UNCHECKED-NEXT:    mov x0, x16
; UNCHECKED-NEXT:    ret
;
; CHECKED-LABEL: test_resign_blend_and_addr:
; CHECKED:       %bb.0:
; CHECKED-NEXT:    mov x16, x0
; CHECKED-ELF-NEXT:    movk x1, #12345, lsl #48
; CHECKED-ELF-NEXT:    autda x16, x1
; CHECKED-DARWIN-NEXT: mov x17, x1
; CHECKED-DARWIN-NEXT: movk x17, #12345, lsl #48
; CHECKED-DARWIN-NEXT: autda x16, x17
; CHECKED-NEXT:    mov x17, x16
; CHECKED-NEXT:    xpacd x17
; CHECKED-NEXT:    cmp x16, x17
; CHECKED-NEXT:    b.eq [[L]]auth_success_2
; CHECKED-NEXT:    mov x16, x17
; CHECKED-NEXT:    b [[L]]resign_end_2
; CHECKED-NEXT:  Lauth_success_2:
; CHECKED-NEXT:    pacdb x16, x2
; CHECKED-NEXT:  Lresign_end_2:
; CHECKED-NEXT:    mov x0, x16
; CHECKED-NEXT:    ret
;
; TRAP-LABEL: test_resign_blend_and_addr:
; TRAP:       %bb.0:
; TRAP-NEXT:    mov x16, x0
; TRAP-ELF-NEXT:    movk x1, #12345, lsl #48
; TRAP-ELF-NEXT:    autda x16, x1
; TRAP-DARWIN-NEXT: mov x17, x1
; TRAP-DARWIN-NEXT: movk x17, #12345, lsl #48
; TRAP-DARWIN-NEXT: autda x16, x17
; TRAP-NEXT:    mov x17, x16
; TRAP-NEXT:    xpacd x17
; TRAP-NEXT:    cmp x16, x17
; TRAP-NEXT:    b.eq [[L]]auth_success_3
; TRAP-NEXT:    brk #0xc472
; TRAP-NEXT:  Lauth_success_3:
; TRAP-NEXT:    pacdb x16, x2
; TRAP-NEXT:    mov x0, x16
; TRAP-NEXT:    ret
  %tmp0 = call i64 @llvm.ptrauth.blend(i64 %arg1, i64 12345)
  %tmp1 = call i64 @llvm.ptrauth.resign(i64 %arg, i32 2, i64 %tmp0, i32 3, i64 %arg2)
  ret i64 %tmp1
}

define i64 @test_auth_too_large_discriminator(i64 %arg, i64 %arg1) {
; UNCHECKED-LABEL:     test_auth_too_large_discriminator:
; UNCHECKED:           %bb.0:
; UNCHECKED-NEXT:        mov w8, #65536
; UNCHECKED-DARWIN-NEXT: bfi x1, x8, #48, #16
; UNCHECKED-DARWIN-NEXT: mov x16, x0
; UNCHECKED-DARWIN-NEXT: autda x16, x1
; UNCHECKED-DARWIN-NEXT: mov x0, x16
; UNCHECKED-ELF-NEXT:    bfi x1, x8, #48, #16
; UNCHECKED-ELF-NEXT:    autda x0, x1
; UNCHECKED-NEXT:        ret
;
; CHECKED-LABEL: test_auth_too_large_discriminator:
; CHECKED:           %bb.0:
; CHECKED-NEXT:        mov w8, #65536
; CHECKED-DARWIN-NEXT: bfi x1, x8, #48, #16
; CHECKED-DARWIN-NEXT: mov x16, x0
; CHECKED-DARWIN-NEXT: autda x16, x1
; CHECKED-DARWIN-NEXT: mov x0, x16
; CHECKED-ELF-NEXT:    bfi x1, x8, #48, #16
; CHECKED-ELF-NEXT:    autda x0, x1
; CHECKED-NEXT:        ret
;
; TRAP-LABEL: test_auth_too_large_discriminator:
; TRAP:           %bb.0:
; TRAP-NEXT:        mov w8, #65536
; TRAP-NEXT:        bfi x1, x8, #48, #16
; TRAP-DARWIN-NEXT: mov x16, x0
; TRAP-DARWIN-NEXT: autda x16, x1
; TRAP-DARWIN-NEXT: mov x17, x16
; TRAP-DARWIN-NEXT: xpacd x17
; TRAP-DARWIN-NEXT: cmp x16, x17
; TRAP-ELF-NEXT:    autda x0, x1
; TRAP-ELF-NEXT:    mov x8, x0
; TRAP-ELF-NEXT:    xpacd x8
; TRAP-ELF-NEXT:    cmp x0, x8
; TRAP-NEXT:        b.eq [[L]]auth_success_4
; TRAP-NEXT:        brk #0xc472
; TRAP-NEXT:      Lauth_success_4:
; TRAP-DARWIN-NEXT: mov x0, x16
; TRAP-NEXT:        ret
  %tmp0 = call i64 @llvm.ptrauth.blend(i64 %arg1, i64 65536)
  %tmp1 = call i64 @llvm.ptrauth.auth(i64 %arg, i32 2, i64 %tmp0)
  ret i64 %tmp1
}

; Without "@earlyclobber $Scratch" constraint on AUTxMxN pseudo, the following
; instruction was fed to AArch64AsmPrinter at -O0
;
;     renamable $x8, dead renamable $x9 = AUTxMxN renamable $x8(tied-def 0), 0, 1, renamable $x9, implicit-def dead $nzcv
;
; resulting in an assertion:
;
;     Assertion `ScratchReg != AddrDisc && "Forbidden to clobber AddrDisc, but have to"
;
define i64 @autxmxn_scratch_is_earlyclobber(i64 %ptr, i64 %arg) {
entry:
  %discr = call i64 @llvm.ptrauth.blend(i64 %arg, i64 1)
  br label %some.bb

some.bb:
  %authed = call i64 @llvm.ptrauth.auth(i64 %ptr, i32 0, i64 %discr)
  br label %some.other.bb

some.other.bb:
  ret i64 %authed
}

define void @test_auth_live_addrdisc(i64 %p, i64 %arg, ptr %out) {
; UNCHECKED-DARWIN-LABEL: test_auth_live_addrdisc:
; UNCHECKED-DARWIN:       %bb.0:
; UNCHECKED-DARWIN-NEXT:    mov     x16, x0
; UNCHECKED-DARWIN-NEXT:    mov     x17, x1
; UNCHECKED-DARWIN-NEXT:    movk    x17, #12345, lsl #48
; UNCHECKED-DARWIN-NEXT:    autda   x16, x17
; UNCHECKED-DARWIN-NEXT:    str     x16, [x2]
; UNCHECKED-DARWIN-NEXT:    str     x1, [x2]
; UNCHECKED-DARWIN-NEXT:    ret
;
; UNCHECKED-ELF-LABEL: test_auth_live_addrdisc:
; UNCHECKED-ELF:       %bb.0:
; UNCHECKED-ELF-NEXT:    mov     x8, x1
; UNCHECKED-ELF-NEXT:    movk    x8, #12345, lsl #48
; UNCHECKED-ELF-NEXT:    autda   x0, x8
; UNCHECKED-ELF-NEXT:    str     x0, [x2]
; UNCHECKED-ELF-NEXT:    str     x1, [x2]
; UNCHECKED-ELF-NEXT:    ret
;
; CHECKED-DARWIN-LABEL: test_auth_live_addrdisc:
; CHECKED-DARWIN:       %bb.0:
; CHECKED-DARWIN-NEXT:    mov     x16, x0
; CHECKED-DARWIN-NEXT:    mov     x17, x1
; CHECKED-DARWIN-NEXT:    movk    x17, #12345, lsl #48
; CHECKED-DARWIN-NEXT:    autda   x16, x17
; CHECKED-DARWIN-NEXT:    str     x16, [x2]
; CHECKED-DARWIN-NEXT:    str     x1, [x2]
; CHECKED-DARWIN-NEXT:    ret
;
; CHECKED-ELF-LABEL: test_auth_live_addrdisc:
; CHECKED-ELF:       %bb.0:
; CHECKED-ELF-NEXT:    mov     x8, x1
; CHECKED-ELF-NEXT:    movk    x8, #12345, lsl #48
; CHECKED-ELF-NEXT:    autda   x0, x8
; CHECKED-ELF-NEXT:    str     x0, [x2]
; CHECKED-ELF-NEXT:    str     x1, [x2]
; CHECKED-ELF-NEXT:    ret
;
; TRAP-DARWIN-LABEL: test_auth_live_addrdisc:
; TRAP-DARWIN:       %bb.0:
; TRAP-DARWIN-NEXT:    mov     x16, x0
; TRAP-DARWIN-NEXT:    mov     x17, x1
; TRAP-DARWIN-NEXT:    movk    x17, #12345, lsl #48
; TRAP-DARWIN-NEXT:    autda   x16, x17
; TRAP-DARWIN-NEXT:    mov     x17, x16
; TRAP-DARWIN-NEXT:    xpacd   x17
; TRAP-DARWIN-NEXT:    cmp     x16, x17
; TRAP-DARWIN-NEXT:    b.eq    [[L]]auth_success_[[N:[0-9]+]]
; TRAP-DARWIN-NEXT:    brk     #0xc472
; TRAP-DARWIN-NEXT:  [[L]]auth_success_[[N]]:
; TRAP-DARWIN-NEXT:    str     x16, [x2]
; TRAP-DARWIN-NEXT:    str     x1, [x2]
; TRAP-DARWIN-NEXT:    ret
;
; TRAP-ELF-LABEL: test_auth_live_addrdisc:
; TRAP-ELF:       %bb.0:
; TRAP-ELF-NEXT:    mov     x8, x1
; TRAP-ELF-NEXT:    movk    x8, #12345, lsl #48
; TRAP-ELF-NEXT:    autda   x0, x8
; TRAP-ELF-NEXT:    mov     x8, x0
; TRAP-ELF-NEXT:    xpacd   x8
; TRAP-ELF-NEXT:    cmp     x0, x8
; TRAP-ELF-NEXT:    b.eq    [[L]]auth_success_[[N:[0-9]+]]
; TRAP-ELF-NEXT:    brk     #0xc472
; TRAP-ELF-NEXT:  [[L]]auth_success_[[N]]:
; TRAP-ELF-NEXT:    str     x0, [x2]
; TRAP-ELF-NEXT:    str     x1, [x2]
; TRAP-ELF-NEXT:    ret
  %auth.disc = call i64 @llvm.ptrauth.blend(i64 %arg, i64 12345)
  %authed = call i64 @llvm.ptrauth.auth(i64 %p, i32 2, i64 %auth.disc)
  store volatile i64 %authed, ptr %out
  store volatile i64 %arg, ptr %out
  ret void
}

define void @test_resign_live_autaddrdisc(i64 %p, i64 %auth.arg, i64 %sign.arg, ptr %out) {
; UNCHECKED-LABEL: test_resign_live_autaddrdisc:
; UNCHECKED:       %bb.0:
; UNCHECKED-NEXT:    mov     x16, x0
; UNCHECKED-NEXT:    mov     x17, x1
; UNCHECKED-NEXT:    movk    x17, #12345, lsl #48
; UNCHECKED-NEXT:    autda   x16, x17
; UNCHECKED-DARWIN-NEXT: mov     x17, x2
; UNCHECKED-DARWIN-NEXT: movk    x17, #54321, lsl #48
; UNCHECKED-DARWIN-NEXT: pacdb   x16, x17
; UNCHECKED-ELF-NEXT:    movk    x2, #54321, lsl #48
; UNCHECKED-ELF-NEXT:    pacdb   x16, x2
; UNCHECKED-NEXT:    str     x16, [x3]
; UNCHECKED-NEXT:    str     x1, [x3]
; UNCHECKED-NEXT:    ret
;
; CHECKED-LABEL: test_resign_live_autaddrdisc:
; CHECKED:       %bb.0:
; CHECKED-NEXT:    mov     x16, x0
; CHECKED-NEXT:    mov     x17, x1
; CHECKED-NEXT:    movk    x17, #12345, lsl #48
; CHECKED-NEXT:    autda   x16, x17
; CHECKED-NEXT:    mov     x17, x16
; CHECKED-NEXT:    xpacd   x17
; CHECKED-NEXT:    cmp     x16, x17
; CHECKED-NEXT:    b.eq    [[L]]auth_success_[[N:[0-9]+]]
; CHECKED-NEXT:    mov     x16, x17
; CHECKED-NEXT:    b       [[L]]resign_end_[[N]]
; CHECKED-NEXT:  [[L]]auth_success_[[N]]:
; CHECKED-DARWIN-NEXT: mov     x17, x2
; CHECKED-DARWIN-NEXT: movk    x17, #54321, lsl #48
; CHECKED-DARWIN-NEXT: pacdb   x16, x17
; CHECKED-ELF-NEXT:    movk    x2, #54321, lsl #48
; CHECKED-ELF-NEXT:    pacdb   x16, x2
; CHECKED-NEXT:  [[L]]resign_end_[[N]]:
; CHECKED-NEXT:    str     x16, [x3]
; CHECKED-NEXT:    str     x1, [x3]
; CHECKED-NEXT:    ret
;
; TRAP-LABEL: test_resign_live_autaddrdisc:
; TRAP:       %bb.0:
; TRAP-NEXT:    mov     x16, x0
; TRAP-NEXT:    mov     x17, x1
; TRAP-NEXT:    movk    x17, #12345, lsl #48
; TRAP-NEXT:    autda   x16, x17
; TRAP-NEXT:    mov     x17, x16
; TRAP-NEXT:    xpacd   x17
; TRAP-NEXT:    cmp     x16, x17
; TRAP-NEXT:    b.eq    [[L]]auth_success_[[N:[0-9]+]]
; TRAP-NEXT:    brk     #0xc472
; TRAP-NEXT:  [[L]]auth_success_[[N]]:
; TRAP-DARWIN-NEXT: mov     x17, x2
; TRAP-DARWIN-NEXT: movk    x17, #54321, lsl #48
; TRAP-DARWIN-NEXT: pacdb   x16, x17
; TRAP-ELF-NEXT:    movk    x2, #54321, lsl #48
; TRAP-ELF-NEXT:    pacdb   x16, x2
; TRAP-NEXT:    str     x16, [x3]
; TRAP-NEXT:    str     x1, [x3]
; TRAP-NEXT:    ret
  %auth.disc = call i64 @llvm.ptrauth.blend(i64 %auth.arg, i64 12345)
  %sign.disc = call i64 @llvm.ptrauth.blend(i64 %sign.arg, i64 54321)
  %resigned = call i64 @llvm.ptrauth.resign(i64 %p, i32 2, i64 %auth.disc, i32 3, i64 %sign.disc)
  store volatile i64 %resigned, ptr %out
  store volatile i64 %auth.arg, ptr %out
  ret void
}

define void @test_resign_live_pacaddrdisc(i64 %p, i64 %auth.arg, i64 %sign.arg, ptr %out) {
; UNCHECKED-LABEL: test_resign_live_pacaddrdisc:
; UNCHECKED:       %bb.0:
; UNCHECKED-NEXT:    mov     x16, x0
; UNCHECKED-DARWIN-NEXT: mov     x17, x1
; UNCHECKED-DARWIN-NEXT: movk    x17, #12345, lsl #48
; UNCHECKED-DARWIN-NEXT: autda   x16, x17
; UNCHECKED-ELF-NEXT:    movk    x1, #12345, lsl #48
; UNCHECKED-ELF-NEXT:    autda   x16, x1
; UNCHECKED-NEXT:    mov     x17, x2
; UNCHECKED-NEXT:    movk    x17, #54321, lsl #48
; UNCHECKED-NEXT:    pacdb   x16, x17
; UNCHECKED-NEXT:    str     x16, [x3]
; UNCHECKED-NEXT:    str     x2, [x3]
; UNCHECKED-NEXT:    ret
;
; CHECKED-LABEL: test_resign_live_pacaddrdisc:
; CHECKED:       %bb.0:
; CHECKED-NEXT:    mov     x16, x0
; CHECKED-DARWIN-NEXT: mov     x17, x1
; CHECKED-DARWIN-NEXT: movk    x17, #12345, lsl #48
; CHECKED-DARWIN-NEXT: autda   x16, x17
; CHECKED-ELF-NEXT:    movk    x1, #12345, lsl #48
; CHECKED-ELF-NEXT:    autda   x16, x1
; CHECKED-NEXT:    mov     x17, x16
; CHECKED-NEXT:    xpacd   x17
; CHECKED-NEXT:    cmp     x16, x17
; CHECKED-NEXT:    b.eq    [[L]]auth_success_[[N:[0-9]+]]
; CHECKED-NEXT:    mov     x16, x17
; CHECKED-NEXT:    b       [[L]]resign_end_[[N]]
; CHECKED-NEXT:  [[L]]auth_success_[[N]]:
; CHECKED-NEXT:    mov     x17, x2
; CHECKED-NEXT:    movk    x17, #54321, lsl #48
; CHECKED-NEXT:    pacdb   x16, x17
; CHECKED-NEXT:  [[L]]resign_end_[[N]]:
; CHECKED-NEXT:    str     x16, [x3]
; CHECKED-NEXT:    str     x2, [x3]
; CHECKED-NEXT:    ret
;
; TRAP-LABEL: test_resign_live_pacaddrdisc:
; TRAP:       %bb.0:
; TRAP-NEXT:    mov     x16, x0
; TRAP-DARWIN-NEXT: mov     x17, x1
; TRAP-DARWIN-NEXT: movk    x17, #12345, lsl #48
; TRAP-DARWIN-NEXT: autda   x16, x17
; TRAP-ELF-NEXT:    movk    x1, #12345, lsl #48
; TRAP-ELF-NEXT:    autda   x16, x1
; TRAP-NEXT:    mov     x17, x16
; TRAP-NEXT:    xpacd   x17
; TRAP-NEXT:    cmp     x16, x17
; TRAP-NEXT:    b.eq    [[L]]auth_success_[[N:[0-9]+]]
; TRAP-NEXT:    brk     #0xc472
; TRAP-NEXT:  [[L]]auth_success_[[N]]:
; TRAP-NEXT:    mov     x17, x2
; TRAP-NEXT:    movk    x17, #54321, lsl #48
; TRAP-NEXT:    pacdb   x16, x17
; TRAP-NEXT:    str     x16, [x3]
; TRAP-NEXT:    str     x2, [x3]
; TRAP-NEXT:    ret
  %auth.disc = call i64 @llvm.ptrauth.blend(i64 %auth.arg, i64 12345)
  %sign.disc = call i64 @llvm.ptrauth.blend(i64 %sign.arg, i64 54321)
  %resigned = call i64 @llvm.ptrauth.resign(i64 %p, i32 2, i64 %auth.disc, i32 3, i64 %sign.disc)
  store volatile i64 %resigned, ptr %out
  store volatile i64 %sign.arg, ptr %out
  ret void
}

; As long as we support raw, non-blended 64-bit discriminators (which might be
; useful for low-level code such as dynamic loaders), the "auth" part of resign
; must not clobber %arg, if its upper bits are later used by the "sign" part.
define i64 @test_resign_aliased_discs_raw_sign_disc(i64 %p, i64 %arg) {
; UNCHECKED-LABEL: test_resign_aliased_discs_raw_sign_disc:
; UNCHECKED:       %bb.0:
; UNCHECKED-NEXT:    mov x16, x0
; UNCHECKED-NEXT:    mov  x17, x1
; UNCHECKED-NEXT:    movk x17, #12345, lsl #48
; UNCHECKED-NEXT:    autda x16, x17
; UNCHECKED-NEXT:    pacdb x16, x1
; UNCHECKED-NEXT:    mov x0, x16
; UNCHECKED-NEXT:    ret
;
; CHECKED-LABEL: test_resign_aliased_discs_raw_sign_disc:
; CHECKED:       %bb.0:
; CHECKED-NEXT:    mov x16, x0
; CHECKED-NEXT:    mov  x17, x1
; CHECKED-NEXT:    movk x17, #12345, lsl #48
; CHECKED-NEXT:    autda x16, x17
; CHECKED-NEXT:    mov x17, x16
; CHECKED-NEXT:    xpacd x17
; CHECKED-NEXT:    cmp x16, x17
; CHECKED-NEXT:    b.eq [[L]]auth_success_[[N:[0-9]+]]
; CHECKED-NEXT:    mov x16, x17
; CHECKED-NEXT:    b [[L]]resign_end_[[N]]
; CHECKED-NEXT:  Lauth_success_[[N]]:
; CHECKED-NEXT:    pacdb x16, x1
; CHECKED-NEXT:  Lresign_end_[[N]]:
; CHECKED-NEXT:    mov x0, x16
; CHECKED-NEXT:    ret
;
; TRAP-LABEL: test_resign_aliased_discs_raw_sign_disc:
; TRAP:       %bb.0:
; TRAP-NEXT:    mov x16, x0
; TRAP-NEXT:    mov  x17, x1
; TRAP-NEXT:    movk x17, #12345, lsl #48
; TRAP-NEXT:    autda x16, x17
; TRAP-NEXT:    mov x17, x16
; TRAP-NEXT:    xpacd x17
; TRAP-NEXT:    cmp x16, x17
; TRAP-NEXT:    b.eq [[L]]auth_success_[[N:[0-9]+]]
; TRAP-NEXT:    brk #0xc472
; TRAP-NEXT:  Lauth_success_[[N]]:
; TRAP-NEXT:    pacdb x16, x1
; TRAP-NEXT:    mov x0, x16
; TRAP-NEXT:    ret
  %auth.disc = call i64 @llvm.ptrauth.blend(i64 %arg, i64 12345)
  %res = call i64 @llvm.ptrauth.resign(i64 %p, i32 2, i64 %auth.disc, i32 3, i64 %arg)
  ret i64 %res
}

; The following are rather obscure corner cases of computing the discriminator
; by blending the pointer-in-question itself as the address modifier.
; For the sake of brevity, just check which registers are used to compute the
; discriminators.
;
; Note: common logic in AArch64AsmPrinter handles $AUTAddrDisc, but $PACAddrDisc
; must be described appropriately in AArch64InstrInfo.td for each pseudo
; instruction to prevent aliasing with the pointer.

define i64 @test_auth_pointer_aliased_with_addrdisc(i64 %p) {
; UNCHECKED-DARWIN-LABEL: test_auth_pointer_aliased_with_addrdisc:
; UNCHECKED-DARWIN:       %bb.0:
; UNCHECKED-DARWIN-NEXT:    mov  x16, x0
; UNCHECKED-DARWIN-NEXT:    mov  x17, x0
; UNCHECKED-DARWIN-NEXT:    movk x17, #12345, lsl #48
; UNCHECKED-DARWIN-NEXT:    autda x16, x17
; UNCHECKED-DARWIN-NEXT:    mov   x0, x16
; UNCHECKED-DARWIN-NEXT:    ret
;
; UNCHECKED-ELF-LABEL: test_auth_pointer_aliased_with_addrdisc:
; UNCHECKED-ELF:       %bb.0:
; UNCHECKED-ELF-NEXT:    mov  x8, x0
; UNCHECKED-ELF-NEXT:    movk x8, #12345, lsl #48
; UNCHECKED-ELF-NEXT:    autda x0, x8
; UNCHECKED-ELF-NEXT:    ret
  %disc = call i64 @llvm.ptrauth.blend(i64 %p, i64 12345)
  %res = call i64 @llvm.ptrauth.auth(i64 %p, i32 2, i64 %disc)
  ret i64 %res
}

define i64 @test_resign_pointer_aliased_with_autaddrdisc(i64 %p, i64 %arg) {
; UNCHECKED-DARWIN-LABEL: test_resign_pointer_aliased_with_autaddrdisc:
; UNCHECKED-DARWIN:       %bb.0:
; UNCHECKED-DARWIN-NEXT:    mov  x16, x0
; UNCHECKED-DARWIN-NEXT:    mov  x17, x0
; UNCHECKED-DARWIN-NEXT:    movk x17, #12345, lsl #48
; UNCHECKED-DARWIN-NEXT:    autda x16, x17
; UNCHECKED-DARWIN-NEXT:    mov  x17, x1
; UNCHECKED-DARWIN-NEXT:    movk x17, #54321, lsl #48
; UNCHECKED-DARWIN-NEXT:    pacdb x16, x17
; UNCHECKED-DARWIN-NEXT:    mov x0, x16
; UNCHECKED-DARWIN-NEXT:    ret
;
; UNCHECKED-ELF-LABEL: test_resign_pointer_aliased_with_autaddrdisc:
; UNCHECKED-ELF:       %bb.0:
; UNCHECKED-ELF-NEXT:    mov  x16, x0
; UNCHECKED-ELF-NEXT:    movk x0, #12345, lsl #48
; UNCHECKED-ELF-NEXT:    autda x16, x0
; UNCHECKED-ELF-NEXT:    movk x1, #54321, lsl #48
; UNCHECKED-ELF-NEXT:    pacdb x16, x1
; UNCHECKED-ELF-NEXT:    mov x0, x16
; UNCHECKED-ELF-NEXT:    ret
  %auth.disc = call i64 @llvm.ptrauth.blend(i64 %p,   i64 12345)
  %sign.disc = call i64 @llvm.ptrauth.blend(i64 %arg, i64 54321)
  %res = call i64 @llvm.ptrauth.resign(i64 %p, i32 2, i64 %auth.disc, i32 3, i64 %sign.disc)
  ret i64 %res
}

define i64 @test_resign_pointer_aliased_with_pacaddrdisc(i64 %p, i64 %arg) {
; UNCHECKED-DARWIN-LABEL: test_resign_pointer_aliased_with_pacaddrdisc:
; UNCHECKED-DARWIN:       %bb.0:
; UNCHECKED-DARWIN-NEXT:    mov x16, x0
; UNCHECKED-DARWIN-NEXT:    mov  x17, x1
; UNCHECKED-DARWIN-NEXT:    movk x17, #12345, lsl #48
; UNCHECKED-DARWIN-NEXT:    autda x16, x17
; UNCHECKED-DARWIN-NEXT:    mov  x17, x0
; UNCHECKED-DARWIN-NEXT:    movk x17, #54321, lsl #48
; UNCHECKED-DARWIN-NEXT:    pacdb x16, x17
; UNCHECKED-DARWIN-NEXT:    mov x0, x16
; UNCHECKED-DARWIN-NEXT:    ret
;

; UNCHECKED-ELF-LABEL: test_resign_pointer_aliased_with_pacaddrdisc:
; UNCHECKED-ELF:       %bb.0:
; UNCHECKED-ELF-NEXT:    mov x16, x0
; UNCHECKED-ELF-NEXT:    movk x1, #12345, lsl #48
; UNCHECKED-ELF-NEXT:    autda x16, x1
; UNCHECKED-ELF-NEXT:    movk x0, #54321, lsl #48
; UNCHECKED-ELF-NEXT:    pacdb x16, x0
; UNCHECKED-ELF-NEXT:    mov x0, x16
; UNCHECKED-ELF-NEXT:    ret
  %auth.disc = call i64 @llvm.ptrauth.blend(i64 %arg, i64 12345)
  %sign.disc = call i64 @llvm.ptrauth.blend(i64 %p,   i64 54321)
  %res = call i64 @llvm.ptrauth.resign(i64 %p, i32 2, i64 %auth.disc, i32 3, i64 %sign.disc)
  ret i64 %res
}

define i64 @test_resign_pointer_aliased_with_both_addrdisc(i64 %p) {
; UNCHECKED-LABEL: test_resign_pointer_aliased_with_both_addrdisc:
; UNCHECKED:       %bb.0:
; UNCHECKED-NEXT:    mov  x16, x0
; UNCHECKED-NEXT:    mov  x17, x0
; UNCHECKED-NEXT:    movk x17, #12345, lsl #48
; UNCHECKED-NEXT:    autda x16, x17
; UNCHECKED-NEXT:    mov  x17, x0
; UNCHECKED-NEXT:    movk x17, #54321, lsl #48
; UNCHECKED-NEXT:    pacdb x16, x17
; UNCHECKED-NEXT:    mov x0, x16
; UNCHECKED-NEXT:    ret
  %auth.disc = call i64 @llvm.ptrauth.blend(i64 %p, i64 12345)
  %sign.disc = call i64 @llvm.ptrauth.blend(i64 %p, i64 54321)
  %res = call i64 @llvm.ptrauth.resign(i64 %p, i32 2, i64 %auth.disc, i32 3, i64 %sign.disc)
  ret i64 %res
}

define i64 @test_autpcpac_pointer_aliased_with_pacaddrdisc(i64 %p, i64 %arg, i64 %auth.pc) {
; UNCHECKED-DARWIN-LABEL: test_autpcpac_pointer_aliased_with_pacaddrdisc:
; UNCHECKED-DARWIN:       %bb.0:
; UNCHECKED-DARWIN-NEXT:    mov  x17, x0
; UNCHECKED-DARWIN-NEXT:    mov  x16, x1
; UNCHECKED-DARWIN-NEXT:    mov  x15, x2
; UNCHECKED-DARWIN-NEXT:    autia171615
; UNCHECKED-DARWIN-NEXT:    mov  x16, x0
; UNCHECKED-DARWIN-NEXT:    movk x16, #54321, lsl #48
; UNCHECKED-DARWIN-NEXT:    pacib x17, x16
; UNCHECKED-DARWIN-NEXT:    mov x0, x17
; UNCHECKED-DARWIN-NEXT:    ret
;
; UNCHECKED-ELF-LABEL: test_autpcpac_pointer_aliased_with_pacaddrdisc:
; UNCHECKED-ELF:       %bb.0:
; UNCHECKED-ELF-NEXT:    mov  x17, x0
; UNCHECKED-ELF-NEXT:    mov  x16, x1
; UNCHECKED-ELF-NEXT:    mov  x15, x2
; UNCHECKED-ELF-NEXT:    autia171615
; UNCHECKED-ELF-NEXT:    movk x0, #54321, lsl #48
; UNCHECKED-ELF-NEXT:    pacib x17, x0
; UNCHECKED-ELF-NEXT:    mov x0, x17
; UNCHECKED-ELF-NEXT:    ret
  %sign.disc = call i64 @llvm.ptrauth.blend(i64 %p, i64 54321)
  %res = call i64 @llvm.ptrauth.auth.with.pc.and.resign(i64 %p, i32 0, i64 %arg, i64 %auth.pc, i32 1, i64 %sign.disc)
  ret i64 %res
}

define i64 @test_autrelloadpac_pointer_aliased_with_pacaddrdisc(i64 %p, i64 %arg) {
; UNCHECKED-DARWIN-LABEL: test_autrelloadpac_pointer_aliased_with_pacaddrdisc:
; UNCHECKED-DARWIN:       %bb.0:
; UNCHECKED-DARWIN-NEXT:    mov  x16, x0
; UNCHECKED-DARWIN-NEXT:    mov  x17, x1
; UNCHECKED-DARWIN-NEXT:    movk x17, #12345, lsl #48
; UNCHECKED-DARWIN-NEXT:    autda x16, x17
; UNCHECKED-DARWIN-NEXT:    ldrsw x17, [x16, #8]!
; UNCHECKED-DARWIN-NEXT:    add   x16, x16, x17
; UNCHECKED-DARWIN-NEXT:    mov  x17, x0
; UNCHECKED-DARWIN-NEXT:    movk x17, #54321, lsl #48
; UNCHECKED-DARWIN-NEXT:    pacdb x16, x17
; UNCHECKED-DARWIN-NEXT:    mov x0, x16
; UNCHECKED-DARWIN-NEXT:    ret
;

; UNCHECKED-ELF-LABEL: test_autrelloadpac_pointer_aliased_with_pacaddrdisc:
; UNCHECKED-ELF:       %bb.0:
; UNCHECKED-ELF-NEXT:    mov  x16, x0
; UNCHECKED-ELF-NEXT:    movk x1, #12345, lsl #48
; UNCHECKED-ELF-NEXT:    autda x16, x1
; UNCHECKED-ELF-NEXT:    ldrsw x17, [x16, #8]!
; UNCHECKED-ELF-NEXT:    add   x16, x16, x17
; UNCHECKED-ELF-NEXT:    movk x0, #54321, lsl #48
; UNCHECKED-ELF-NEXT:    pacdb x16, x0
; UNCHECKED-ELF-NEXT:    mov x0, x16
; UNCHECKED-ELF-NEXT:    ret
  %auth.disc = call i64 @llvm.ptrauth.blend(i64 %arg, i64 12345)
  %sign.disc = call i64 @llvm.ptrauth.blend(i64 %p,   i64 54321)
  %res = call i64 @llvm.ptrauth.resign.load.relative(i64 %p, i32 2, i64 %auth.disc, i32 3, i64 %sign.disc, i64 8)
  ret i64 %res
}

declare i64 @llvm.ptrauth.auth(i64, i32, i64)
declare i64 @llvm.ptrauth.resign(i64, i32, i64, i32, i64)
declare i64 @llvm.ptrauth.blend(i64, i64)
