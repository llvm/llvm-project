; RUN: rm -rf %t && split-file %s %t && cd %t
; RUN: cat common.ll authflag.ll > auth.ll
; RUN: cat common.ll authflag-bb.ll > auth-bb.ll

; RUN: llc -mtriple=aarch64-none-linux-gnu -global-isel=0 -fast-isel=0         -relocation-model=pic \
; RUN:   -mattr=+pauth -mattr=+fpac -o - auth.ll | FileCheck --check-prefixes=CHECK,NOTRAP,NOTRAPA %s
; RUN: llc -mtriple=aarch64-none-linux-gnu -global-isel=0 -fast-isel=0         -relocation-model=pic \
; RUN:   -mattr=+pauth              -o - auth.ll | FileCheck --check-prefixes=CHECK,TRAP,TRAPA %s

; RUN: llc -mtriple=aarch64-none-linux-gnu -global-isel=0 -fast-isel=1         -relocation-model=pic \
; RUN:   -mattr=+pauth -mattr=+fpac -o - auth.ll | FileCheck --check-prefixes=CHECK,NOTRAP,NOTRAPA %s
; RUN: llc -mtriple=aarch64-none-linux-gnu -global-isel=0 -fast-isel=1         -relocation-model=pic \
; RUN:   -mattr=+pauth              -o - auth.ll | FileCheck --check-prefixes=CHECK,TRAP,TRAPA %s

; RUN: llc -mtriple=aarch64-none-linux-gnu -global-isel=1 -global-isel-abort=1 -relocation-model=pic \
; RUN:   -mattr=+pauth -mattr=+fpac -o - auth.ll | FileCheck --check-prefixes=CHECK,NOTRAP,NOTRAPA %s
; RUN: llc -mtriple=aarch64-none-linux-gnu -global-isel=1 -global-isel-abort=1 -relocation-model=pic \
; RUN:   -mattr=+pauth              -o - auth.ll | FileCheck --check-prefixes=CHECK,TRAP,TRAPA %s

; RUN: llc -mtriple=aarch64-none-linux-gnu -code-model=tiny -mattr=+pauth -mattr=+fpac -o - auth.ll | \
; RUN:   FileCheck --check-prefixes=CHECK-TINY,NOTRAP-TINY,NOTRAP-TINYA %s
; RUN: llc -mtriple=aarch64-none-linux-gnu -code-model=tiny -mattr=+pauth              -o - auth.ll | \
; RUN:   FileCheck --check-prefixes=CHECK-TINY,TRAP-TINY,TRAP-TINYA %s

; RUN: llc -mtriple=aarch64-none-linux-gnu -global-isel=0 -fast-isel=0         -relocation-model=pic \
; RUN:   -mattr=+pauth -mattr=+fpac -o - auth-bb.ll | FileCheck --check-prefixes=CHECK,NOTRAP,NOTRAPB %s
; RUN: llc -mtriple=aarch64-none-linux-gnu -global-isel=0 -fast-isel=0         -relocation-model=pic \
; RUN:   -mattr=+pauth              -o - auth-bb.ll | FileCheck --check-prefixes=CHECK,TRAP,TRAPB %s

; RUN: llc -mtriple=aarch64-none-linux-gnu -global-isel=0 -fast-isel=1         -relocation-model=pic \
; RUN:   -mattr=+pauth -mattr=+fpac -o - auth-bb.ll | FileCheck --check-prefixes=CHECK,NOTRAP,NOTRAPB %s
; RUN: llc -mtriple=aarch64-none-linux-gnu -global-isel=0 -fast-isel=1         -relocation-model=pic \
; RUN:   -mattr=+pauth              -o - auth-bb.ll | FileCheck --check-prefixes=CHECK,TRAP,TRAPB %s

; RUN: llc -mtriple=aarch64-none-linux-gnu -global-isel=1 -global-isel-abort=1 -relocation-model=pic \
; RUN:   -mattr=+pauth -mattr=+fpac -o - auth-bb.ll | FileCheck --check-prefixes=CHECK,NOTRAP,NOTRAPB %s
; RUN: llc -mtriple=aarch64-none-linux-gnu -global-isel=1 -global-isel-abort=1 -relocation-model=pic \
; RUN:   -mattr=+pauth              -o - auth-bb.ll | FileCheck --check-prefixes=CHECK,TRAP,TRAPB %s

; RUN: llc -mtriple=aarch64-none-linux-gnu -code-model=tiny -mattr=+pauth -mattr=+fpac -o - auth-bb.ll | \
; RUN:   FileCheck --check-prefixes=CHECK-TINY,NOTRAP-TINY,NOTRAP-TINYB %s
; RUN: llc -mtriple=aarch64-none-linux-gnu -code-model=tiny -mattr=+pauth              -o - auth-bb.ll | \
; RUN:   FileCheck --check-prefixes=CHECK-TINY,TRAP-TINY,TRAP-TINYB %s

;; Note: for FastISel, we fall back to SelectionDAG

;--- common.ll
declare extern_weak dso_local i32 @var()

define ptr @foo() {
; The usual ADRP/ADD pair can't be used for a weak reference because it must
; evaluate to 0 if the symbol is undefined. We use a GOT entry for PIC
; otherwise a litpool entry.
  ret ptr @var

; CHECK-LABEL: foo:
; CHECK:         adrp  x17, :got_auth:var
; CHECK-NEXT:    add   x17, x17, :got_auth_lo12:var
; NOTRAP-NEXT:   ldr   x0,  [x17]
; NOTRAP-NEXT:   cbz   x0,  .Lundef_weak0
; NOTRAPA-NEXT:  autia x0,  x17
; NOTRAPB-NEXT:  autib x0,  x17
; TRAP-NEXT:     ldr   x16, [x17]
; TRAP-NEXT:     cbz   x16, .Lundef_weak0
; TRAPA-NEXT:    autia x16, x17
; TRAPB-NEXT:    autib x16, x17
; CHECK-NEXT:  .Lundef_weak0:
; TRAP-NEXT:     mov   x17, x16
; TRAP-NEXT:     xpaci x17
; TRAP-NEXT:     cmp   x16, x17
; TRAP-NEXT:     b.eq  .Lauth_success_0
; TRAPA-NEXT:    brk   #0xc470
; TRAPB-NEXT:    brk   #0xc471
; TRAP-NEXT:   .Lauth_success_0:
; TRAP-NEXT:     mov   x0,  x16
; CHECK-NEXT:    ret

; CHECK-TINY-LABEL: foo:
; CHECK-TINY:         adr   x17, :got_auth:var
; NOTRAP-TINY-NEXT:   ldr   x0,  [x17]
; NOTRAP-TINY-NEXT:   cbz   x0,  .Lundef_weak0
; NOTRAP-TINYA-NEXT:  autia x0,  x17
; NOTRAP-TINYB-NEXT:  autib x0,  x17
; TRAP-TINY-NEXT:     ldr   x16, [x17]
; TRAP-TINY-NEXT:     cbz   x16, .Lundef_weak0
; TRAP-TINYA-NEXT:    autia x16, x17
; TRAP-TINYB-NEXT:    autib x16, x17
; CHECK-TINY-NEXT:  .Lundef_weak0:
; TRAP-TINY-NEXT:     mov   x17, x16
; TRAP-TINY-NEXT:     xpaci x17
; TRAP-TINY-NEXT:     cmp   x16, x17
; TRAP-TINY-NEXT:     b.eq  .Lauth_success_0
; TRAP-TINYA-NEXT:    brk   #0xc470
; TRAP-TINYB-NEXT:    brk   #0xc471
; TRAP-TINY-NEXT:   .Lauth_success_0:
; TRAP-TINY-NEXT:     mov   x0,  x16
; CHECK-TINY-NEXT:    ret
}

@arr_var = extern_weak global [10 x i32]

define ptr @bar() {
  %addr = getelementptr [10 x i32], ptr @arr_var, i32 0, i32 5
  ret ptr %addr

; CHECK-LABEL: bar:
; CHECK:         adrp  x17, :got_auth:arr_var
; CHECK-NEXT:    add   x17, x17, :got_auth_lo12:arr_var
; NOTRAP-NEXT:   ldr   x8,  [x17]
; NOTRAP-NEXT:   cbz   x8,  .Lundef_weak1
; NOTRAPA-NEXT:  autda x8,  x17
; NOTRAPB-NEXT:  autdb x8,  x17
; TRAP-NEXT:     ldr   x16, [x17]
; TRAP-NEXT:     cbz   x16, .Lundef_weak1
; TRAPA-NEXT:    autda x16, x17
; TRAPB-NEXT:    autdb x16, x17
; CHECK-NEXT:  .Lundef_weak1:
; TRAP-NEXT:     mov   x17, x16
; TRAP-NEXT:     xpacd x17
; TRAP-NEXT:     cmp   x16, x17
; TRAP-NEXT:     b.eq  .Lauth_success_1
; TRAPA-NEXT:    brk   #0xc472
; TRAPB-NEXT:    brk   #0xc473
; TRAP-NEXT:   .Lauth_success_1:
; TRAP-NEXT:     mov   x8,  x16
; CHECK-NEXT:    add   x0,  x8, #20
; CHECK-NEXT:    ret

; CHECK-TINY-LABEL: bar:
; CHECK-TINY:         adr   x17, :got_auth:arr_var
; NOTRAP-TINY-NEXT:   ldr   x8,  [x17]
; NOTRAP-TINY-NEXT:   cbz   x8,  .Lundef_weak1
; NOTRAP-TINYA-NEXT:  autda x8,  x17
; NOTRAP-TINYB-NEXT:  autdb x8,  x17
; TRAP-TINY-NEXT:     ldr   x16, [x17]
; TRAP-TINY-NEXT:     cbz   x16, .Lundef_weak1
; TRAP-TINYA-NEXT:    autda x16, x17
; TRAP-TINYB-NEXT:    autdb x16, x17
; CHECK-TINY-NEXT:  .Lundef_weak1:
; TRAP-TINY-NEXT:     mov   x17, x16
; TRAP-TINY-NEXT:     xpacd x17
; TRAP-TINY-NEXT:     cmp   x16, x17
; TRAP-TINY-NEXT:     b.eq  .Lauth_success_1
; TRAP-TINYA-NEXT:    brk   #0xc472
; TRAP-TINYB-NEXT:    brk   #0xc473
; TRAP-TINY-NEXT:   .Lauth_success_1:
; TRAP-TINY-NEXT:     mov   x8,  x16
; CHECK-TINY-NEXT:    add   x0,  x8, #20
; CHECK-TINY-NEXT:    ret
}

;--- authflag.ll
!llvm.module.flags = !{!0}
!0 = !{i32 8, !"ptrauth-elf-got", i32 1}

;--- authflag-bb.ll
!llvm.module.flags = !{!0, !1}
!0 = !{i32 8, !"ptrauth-elf-got", i32 1}
!1 = !{i32 1, !"ptrauth-keys", !"ibdbibdb"}
