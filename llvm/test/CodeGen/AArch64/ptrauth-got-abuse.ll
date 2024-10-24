; RUN: llc -mtriple=aarch64-none-linux-gnu -asm-verbose=false -global-isel=0 -fast-isel=0         \
; RUN:   -relocation-model=pic -mattr=+pauth -mattr=+fpac -o - %s | FileCheck %s
; RUN: llc -mtriple=aarch64-none-linux-gnu -asm-verbose=false -global-isel=0 -fast-isel=1         \
; RUN:   -relocation-model=pic -mattr=+pauth -mattr=+fpac -o - %s | FileCheck %s
; RUN: llc -mtriple=aarch64-none-linux-gnu -asm-verbose=false -global-isel=1 -global-isel-abort=1 \
; RUN:   -relocation-model=pic -mattr=+pauth -mattr=+fpac -o - %s | FileCheck %s

; RUN: llc -mtriple=aarch64-none-linux-gnu -asm-verbose=false -global-isel=0 -fast-isel=0         \
; RUN:   -relocation-model=pic -filetype=obj -mattr=+pauth -o /dev/null %s
; RUN: llc -mtriple=aarch64-none-linux-gnu -asm-verbose=false -global-isel=0 -fast-isel=1         \
; RUN:   -relocation-model=pic -filetype=obj -mattr=+pauth -o /dev/null %s
; RUN: llc -mtriple=aarch64-none-linux-gnu -asm-verbose=false -global-isel=1 -global-isel-abort=1 \
; RUN:   -relocation-model=pic -filetype=obj -mattr=+pauth -o /dev/null %s

;; Note: for FastISel, we fall back to SelectionDAG

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
; CHECK:         adrp  x16, :got_auth:func
; CHECK-NEXT:    add   x16, x16, :got_auth_lo12:func
; CHECK-NEXT:    ldr   x[[TMP0:[0-9]+]], [x16]
; CHECK-NEXT:    autia x[[TMP0]], x16

  call void @consume(i32 ptrtoint (ptr @alias_func to i32))
; CHECK:         adrp  x16, :got_auth:alias_func
; CHECK-NEXT:    add   x16, x16, :got_auth_lo12:alias_func
; CHECK-NEXT:    ldr   x[[TMP1:[0-9]+]], [x16]
; CHECK-NEXT:    autia x[[TMP1]], x16

  call void @consume(i32 ptrtoint (ptr @alias_global to i32))
; CHECK:         adrp  x16, :got_auth:alias_global
; CHECK-NEXT:    add   x16, x16, :got_auth_lo12:alias_global
; CHECK-NEXT:    ldr   x[[TMP2:[0-9]+]], [x16]
; CHECK-NEXT:    autda x[[TMP2]], x16

  ret void
}

!llvm.module.flags = !{!0}
!0 = !{i32 8, !"ptrauth-elf-got", i32 1}
