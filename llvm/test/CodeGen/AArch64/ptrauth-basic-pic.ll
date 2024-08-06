; RUN: llc -mtriple=aarch64-linux-gnu -global-isel=0 -fast-isel=0         -verify-machineinstrs \
; RUN:   -relocation-model=pic -mattr=+pauth %s -o - | FileCheck %s
; RUN: llc -mtriple=aarch64-linux-gnu -global-isel=0 -fast-isel=1         -verify-machineinstrs \
; RUN:   -relocation-model=pic -mattr=+pauth %s -o - | FileCheck %s
; RUN: llc -mtriple=aarch64-linux-gnu -global-isel=1 -global-isel-abort=1 -verify-machineinstrs \
; RUN:   -relocation-model=pic -mattr=+pauth %s -o - | FileCheck %s

;; Note: for FastISel, we fall back to SelectionDAG

@var = global i32 0

define i32 @get_globalvar() {
; CHECK-LABEL: get_globalvar:
; CHECK:         adrp x[[GOT:[0-9]+]], :got_auth:var
; CHECK-NEXT:    add x[[GOT]], x[[GOT]], :got_auth_lo12:var
; CHECK-NEXT:    ldr x[[SYM:[0-9]+]], [x[[GOT]]]
; CHECK-NEXT:    autda x[[SYM]], x[[GOT]]
; CHECK-NEXT:    ldr w0, [x[[SYM]]]

  %val = load i32, ptr @var
  ret i32 %val
}

define ptr @get_globalvaraddr() {
; CHECK-LABEL: get_globalvaraddr:
; CHECK:         adrp x[[GOT:[0-9]+]], :got_auth:var
; CHECK-NEXT:    add x[[GOT]], x[[GOT]], :got_auth_lo12:var
; CHECK-NEXT:    ldr x0, [x[[GOT]]]
; CHECK-NEXT:    autda x0, x[[GOT]]

  %val = load i32, ptr @var
  ret ptr @var
}

declare i32 @foo()

define ptr @resign_globalfunc() {
; CHECK-LABEL: resign_globalfunc:
; CHECK:         adrp     x17, :got_auth:foo
; CHECK-NEXT:    add      x17, x17, :got_auth_lo12:foo
; CHECK-NEXT:    ldr      x16, [x17]
; CHECK-NEXT:    autia    x16, x17
; CHECK-NEXT:    mov x17, #42
; CHECK-NEXT:    pacia    x16, x17
; CHECK-NEXT:    mov      x0, x16
; CHECK-NEXT:    ret

  ret ptr ptrauth (ptr @foo, i32 0, i64 42)
}

define ptr @resign_globalvar() {
; CHECK-LABEL: resign_globalvar:
; CHECK:         adrp     x17, :got_auth:var
; CHECK-NEXT:    add      x17, x17, :got_auth_lo12:var
; CHECK-NEXT:    ldr      x16, [x17]
; CHECK-NEXT:    autda    x16, x17
; CHECK-NEXT:    mov x17, #43
; CHECK-NEXT:    pacdb    x16, x17
; CHECK-NEXT:    mov      x0, x16
; CHECK-NEXT:    ret

  ret ptr ptrauth (ptr @var, i32 3, i64 43)
}

define ptr @resign_globalvar_offset() {
; CHECK-LABEL: resign_globalvar_offset:
; CHECK:         adrp  x17, :got_auth:var
; CHECK-NEXT:    add   x17, x17, :got_auth_lo12:var
; CHECK-NEXT:    ldr   x16, [x17]
; CHECK-NEXT:    autda x16, x17
; CHECK-NEXT:    add   x16, x16, #16
; CHECK-NEXT:    mov   x17, #44
; CHECK-NEXT:    pacda x16, x17
; CHECK-NEXT:    mov   x0, x16
; CHECK-NEXT:    ret

  ret ptr ptrauth (ptr getelementptr (i8, ptr @var, i64 16), i32 2, i64 44)
}

!llvm.module.flags = !{!0, !1}
!0 = !{i32 1, !"aarch64-elf-pauthabi-platform", i32 268435458}
!1 = !{i32 1, !"aarch64-elf-pauthabi-version", i32 256}
