; RUN: llc -mtriple=aarch64-none-linux-gnu -global-isel=0 -fast-isel=0         -relocation-model=pic -mattr=+pauth -o - %s | FileCheck %s
; RUN: llc -mtriple=aarch64-none-linux-gnu -global-isel=0 -fast-isel=1         -relocation-model=pic -mattr=+pauth -o - %s | FileCheck %s
; RUN: llc -mtriple=aarch64-none-linux-gnu -global-isel=1 -global-isel-abort=1 -relocation-model=pic -mattr=+pauth -o - %s | FileCheck %s

;; Note: for FastISel, we fall back to SelectionDAG

declare extern_weak dso_local i32 @var()

define ptr @foo() {
; The usual ADRP/ADD pair can't be used for a weak reference because it must
; evaluate to 0 if the symbol is undefined. We use a GOT entry for PIC
; otherwise a litpool entry.
  ret ptr @var

; CHECK:      adrp x[[ADDRHI:[0-9]+]], :got_auth:var
; CHECK-NEXT: add x[[ADDRHI]], x[[ADDRHI]], :got_auth_lo12:var
; CHECK-NEXT: ldr x0, [x[[ADDRHI]]]
; CHECK-NEXT: autia x0, x[[ADDRHI]]
}

@arr_var = extern_weak global [10 x i32]

define ptr @bar() {
  %addr = getelementptr [10 x i32], ptr @arr_var, i32 0, i32 5

; CHECK:      adrp x[[ADDRHI:[0-9]+]], :got_auth:arr_var
; CHECK-NEXT: add x[[ADDRHI]], x[[ADDRHI]], :got_auth_lo12:arr_var
; CHECK-NEXT: ldr [[BASE:x[0-9]+]], [x[[ADDRHI]]]
; CHECK-NEXT: autda [[BASE]], x[[ADDRHI]]
; CHECK-NEXT: add x0, [[BASE]], #20
  ret ptr %addr
}

!llvm.module.flags = !{!0, !1}
!0 = !{i32 1, !"aarch64-elf-pauthabi-platform", i32 268435458}
!1 = !{i32 1, !"aarch64-elf-pauthabi-version", i32 256}
