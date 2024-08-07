; RUN: llc -mtriple=aarch64-none-linux-gnu -asm-verbose=false -global-isel=0 -fast-isel=0         -relocation-model=pic -mattr=+pauth -o - %s | FileCheck %s
; RUN: llc -mtriple=aarch64-none-linux-gnu -asm-verbose=false -global-isel=0 -fast-isel=1         -relocation-model=pic -mattr=+pauth -o - %s | FileCheck %s
; RUN: llc -mtriple=aarch64-none-linux-gnu -asm-verbose=false -global-isel=1 -global-isel-abort=1 -relocation-model=pic -mattr=+pauth -o - %s | FileCheck %s
; RUN: llc -mtriple=aarch64-none-linux-gnu -asm-verbose=false -global-isel=0 -fast-isel=0         -relocation-model=pic -filetype=obj -mattr=+pauth -o /dev/null %s
; RUN: llc -mtriple=aarch64-none-linux-gnu -asm-verbose=false -global-isel=0 -fast-isel=1         -relocation-model=pic -filetype=obj -mattr=+pauth -o /dev/null %s
; RUN: llc -mtriple=aarch64-none-linux-gnu -asm-verbose=false -global-isel=1 -global-isel-abort=1 -relocation-model=pic -filetype=obj -mattr=+pauth -o /dev/null %s

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
; CHECK:      adrp x[[ADDRHI:[0-9]+]], :got_auth:func
; CHECK-NEXT: add x[[ADDRHI]], x[[ADDRHI]], :got_auth_lo12:func
; CHECK-NEXT: ldr x[[SYM:[0-9]+]], [x[[ADDRHI]]]
; CHECK-NEXT: autia x[[SYM:[0-9]+]], x[[ADDRHI]]
  call void @consume(i32 ptrtoint (ptr @alias_func to i32))
; CHECK:      adrp x[[ADDRHI:[0-9]+]], :got_auth:alias_func
; CHECK-NEXT: add x[[ADDRHI]], x[[ADDRHI]], :got_auth_lo12:alias_func
; CHECK-NEXT: ldr x[[SYM:[0-9]+]], [x[[ADDRHI]]]
; CHECK-NEXT: autia x[[SYM:[0-9]+]], x[[ADDRHI]]
  call void @consume(i32 ptrtoint (ptr @alias_global to i32))
; CHECK:      adrp x[[ADDRHI:[0-9]+]], :got_auth:alias_global
; CHECK-NEXT: add x[[ADDRHI]], x[[ADDRHI]], :got_auth_lo12:alias_global
; CHECK-NEXT: ldr x[[SYM:[0-9]+]], [x[[ADDRHI]]]
; CHECK-NEXT: autda x[[SYM:[0-9]+]], x[[ADDRHI]]
  ret void
}

!llvm.module.flags = !{!0, !1}
!0 = !{i32 1, !"aarch64-elf-pauthabi-platform", i32 268435458}
!1 = !{i32 1, !"aarch64-elf-pauthabi-version", i32 256}
