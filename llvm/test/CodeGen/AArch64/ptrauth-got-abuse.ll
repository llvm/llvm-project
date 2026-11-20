; RUN: rm -rf %t && split-file %s %t && cd %t
; RUN: cat common.ll authflag.ll > auth.ll
; RUN: cat common.ll authflag-bb.ll > auth-bb.ll

; RUN: llc -mtriple=aarch64-none-linux-gnu -asm-verbose=false -global-isel=0 -fast-isel=0         \
; RUN:   -relocation-model=pic -mattr=+pauth -mattr=+fpac -o - auth.ll | FileCheck --check-prefixes=CHECK,NOTRAP,NOTRAPA %s
; RUN: llc -mtriple=aarch64-none-linux-gnu -asm-verbose=false -global-isel=0 -fast-isel=0         \
; RUN:   -relocation-model=pic -mattr=+pauth              -o - auth.ll | FileCheck --check-prefixes=CHECK,TRAP,TRAPA %s

; RUN: llc -mtriple=aarch64-none-linux-gnu -asm-verbose=false -global-isel=0 -fast-isel=1         \
; RUN:   -relocation-model=pic -mattr=+pauth -mattr=+fpac -o - auth.ll | FileCheck --check-prefixes=CHECK,NOTRAP,NOTRAPA %s
; RUN: llc -mtriple=aarch64-none-linux-gnu -asm-verbose=false -global-isel=0 -fast-isel=1         \
; RUN:   -relocation-model=pic -mattr=+pauth              -o - auth.ll | FileCheck --check-prefixes=CHECK,TRAP,TRAPA %s

; RUN: llc -mtriple=aarch64-none-linux-gnu -asm-verbose=false -global-isel=1 -global-isel-abort=1 \
; RUN:   -relocation-model=pic -mattr=+pauth -mattr=+fpac -o - auth.ll | FileCheck --check-prefixes=CHECK,NOTRAP,NOTRAPA %s
; RUN: llc -mtriple=aarch64-none-linux-gnu -asm-verbose=false -global-isel=1 -global-isel-abort=1 \
; RUN:   -relocation-model=pic -mattr=+pauth              -o - auth.ll | FileCheck --check-prefixes=CHECK,TRAP,TRAPA %s

; RUN: llc -mtriple=aarch64-none-linux-gnu -asm-verbose=false -global-isel=0 -fast-isel=0         \
; RUN:   -relocation-model=pic -filetype=obj -mattr=+pauth -o /dev/null auth.ll
; RUN: llc -mtriple=aarch64-none-linux-gnu -asm-verbose=false -global-isel=0 -fast-isel=1         \
; RUN:   -relocation-model=pic -filetype=obj -mattr=+pauth -o /dev/null auth.ll
; RUN: llc -mtriple=aarch64-none-linux-gnu -asm-verbose=false -global-isel=1 -global-isel-abort=1 \
; RUN:   -relocation-model=pic -filetype=obj -mattr=+pauth -o /dev/null auth.ll

; RUN: llc -mtriple=aarch64-none-linux-gnu -asm-verbose=false -global-isel=0 -fast-isel=0         \
; RUN:   -relocation-model=pic -mattr=+pauth -mattr=+fpac -o - auth-bb.ll | FileCheck --check-prefixes=CHECK,NOTRAP,NOTRAPB %s
; RUN: llc -mtriple=aarch64-none-linux-gnu -asm-verbose=false -global-isel=0 -fast-isel=0         \
; RUN:   -relocation-model=pic -mattr=+pauth              -o - auth-bb.ll | FileCheck --check-prefixes=CHECK,TRAP,TRAPB %s

; RUN: llc -mtriple=aarch64-none-linux-gnu -asm-verbose=false -global-isel=0 -fast-isel=1         \
; RUN:   -relocation-model=pic -mattr=+pauth -mattr=+fpac -o - auth-bb.ll | FileCheck --check-prefixes=CHECK,NOTRAP,NOTRAPB %s
; RUN: llc -mtriple=aarch64-none-linux-gnu -asm-verbose=false -global-isel=0 -fast-isel=1         \
; RUN:   -relocation-model=pic -mattr=+pauth              -o - auth-bb.ll | FileCheck --check-prefixes=CHECK,TRAP,TRAPB %s

; RUN: llc -mtriple=aarch64-none-linux-gnu -asm-verbose=false -global-isel=1 -global-isel-abort=1 \
; RUN:   -relocation-model=pic -mattr=+pauth -mattr=+fpac -o - auth-bb.ll | FileCheck --check-prefixes=CHECK,NOTRAP,NOTRAPB %s
; RUN: llc -mtriple=aarch64-none-linux-gnu -asm-verbose=false -global-isel=1 -global-isel-abort=1 \
; RUN:   -relocation-model=pic -mattr=+pauth              -o - auth-bb.ll | FileCheck --check-prefixes=CHECK,TRAP,TRAPB %s

; RUN: llc -mtriple=aarch64-none-linux-gnu -asm-verbose=false -global-isel=0 -fast-isel=0         \
; RUN:   -relocation-model=pic -filetype=obj -mattr=+pauth -o /dev/null auth-bb.ll
; RUN: llc -mtriple=aarch64-none-linux-gnu -asm-verbose=false -global-isel=0 -fast-isel=1         \
; RUN:   -relocation-model=pic -filetype=obj -mattr=+pauth -o /dev/null auth-bb.ll
; RUN: llc -mtriple=aarch64-none-linux-gnu -asm-verbose=false -global-isel=1 -global-isel-abort=1 \
; RUN:   -relocation-model=pic -filetype=obj -mattr=+pauth -o /dev/null auth-bb.ll

;; Note: for FastISel, we fall back to SelectionDAG

;--- common.ll
declare void @consume(i32)
declare void @func()

define void @aliasee_func() {
  ret void
}
@alias_func = alias void (), ptr @aliasee_func

@aliasee_global = global i32 42
@alias_global = alias i32, ptr @aliasee_global

define void @foo() nounwind {
; CHECK-LABEL: foo:
entry:
  call void @consume(i32 ptrtoint (ptr @func to i32))
; CHECK:         adrp  x17, :got_auth:func
; CHECK-NEXT:    add   x17, x17, :got_auth_lo12:func
; NOTRAP-NEXT:   ldr   x[[TMP0:[0-9]+]], [x17]
; NOTRAPA-NEXT:  autia x[[TMP0]], x17
; NOTRAPB-NEXT:  autib x[[TMP0]], x17
; TRAP-NEXT:     ldr   x16, [x17]
; TRAPA-NEXT:    autia x16, x17
; TRAPB-NEXT:    autib x16, x17
; TRAP-NEXT:     mov   x17, x16
; TRAP-NEXT:     xpaci x17
; TRAP-NEXT:     cmp   x16, x17
; TRAP-NEXT:     b.eq  .Lauth_success_0
; TRAPA-NEXT:    brk   #0xc470
; TRAPB-NEXT:    brk   #0xc471
; TRAP-NEXT:   .Lauth_success_0:
; TRAP-NEXT:     mov   x[[TMP0:[0-9]+]], x16

  call void @consume(i32 ptrtoint (ptr @alias_func to i32))
; CHECK:         adrp  x17, :got_auth:alias_func
; CHECK-NEXT:    add   x17, x17, :got_auth_lo12:alias_func
; NOTRAP-NEXT:   ldr   x[[TMP1:[0-9]+]], [x17]
; NOTRAPA-NEXT:  autia x[[TMP1]], x17
; NOTRAPB-NEXT:  autib x[[TMP1]], x17
; TRAP-NEXT:     ldr   x16, [x17]
; TRAPA-NEXT:    autia x16, x17
; TRAPB-NEXT:    autib x16, x17
; TRAP-NEXT:     mov   x17, x16
; TRAP-NEXT:     xpaci x17
; TRAP-NEXT:     cmp   x16, x17
; TRAP-NEXT:     b.eq  .Lauth_success_1
; TRAPA-NEXT:    brk   #0xc470
; TRAPB-NEXT:    brk   #0xc471
; TRAP-NEXT:   .Lauth_success_1:
; TRAP-NEXT:     mov   x[[TMP1:[0-9]+]], x16

  call void @consume(i32 ptrtoint (ptr @alias_global to i32))
; CHECK:         adrp  x17, :got_auth:alias_global
; CHECK-NEXT:    add   x17, x17, :got_auth_lo12:alias_global
; NOTRAP-NEXT:   ldr   x[[TMP2:[0-9]+]], [x17]
; NOTRAPA-NEXT:  autda x[[TMP2]], x17
; NOTRAPB-NEXT:  autdb x[[TMP2]], x17
; TRAP-NEXT:     ldr   x16, [x17]
; TRAPA-NEXT:    autda x16, x17
; TRAPB-NEXT:    autdb x16, x17
; TRAP-NEXT:     mov   x17, x16
; TRAP-NEXT:     xpacd x17
; TRAP-NEXT:     cmp   x16, x17
; TRAP-NEXT:     b.eq  .Lauth_success_2
; TRAPA-NEXT:    brk   #0xc472
; TRAPB-NEXT:    brk   #0xc473
; TRAP-NEXT:   .Lauth_success_2:
; TRAP-NEXT:     mov   x[[TMP2:[0-9]+]], x16

  ret void
}

;--- authflag.ll
!llvm.module.flags = !{!0}
!0 = !{i32 8, !"ptrauth-elf-got", i32 1}

;--- authflag-bb.ll
!llvm.module.flags = !{!0, !1}
!0 = !{i32 8, !"ptrauth-elf-got", i32 1}
!1 = !{i32 1, !"ptrauth-keys", !"ibdbibdb"}
