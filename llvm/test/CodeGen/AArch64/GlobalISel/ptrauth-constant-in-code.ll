; RUN: rm -rf %t && split-file %s %t && cd %t

;--- err1.ll

; RUN: not --crash llc < err1.ll -mtriple aarch64-elf -mattr=+pauth \
; RUN:   -global-isel=1 -verify-machineinstrs -global-isel-abort=1 2>&1 | \
; RUN:   FileCheck --check-prefix=ERR1 %s

@g = external global i32

define ptr @foo() {
; ERR1: LLVM ERROR: key in ptrauth global out of range [0, 3]
  ret ptr ptrauth (ptr @g, i32 4)
}

;--- err2.ll

; RUN: not --crash llc < err2.ll -mtriple aarch64-elf -mattr=+pauth \
; RUN:   -global-isel=1 -verify-machineinstrs -global-isel-abort=1 2>&1 | \
; RUN:   FileCheck --check-prefix=ERR2 %s

@g = external global i32

define ptr @foo() {
; ERR2: LLVM ERROR: constant discriminator in ptrauth global out of range [0, 0xffff]
  ret ptr ptrauth (ptr @g, i32 2, i64 65536)
}

;--- err3.ll

; RUN: not --crash llc < err3.ll -mtriple aarch64-elf -mattr=+pauth \
; RUN:   -global-isel=1 -verify-machineinstrs -global-isel-abort=1 2>&1 | \
; RUN:   FileCheck --check-prefix=ERR3 %s

@g_weak = extern_weak global i32

define ptr @foo() {
; ERR3: LLVM ERROR: unsupported non-zero offset in weak ptrauth global reference
  ret ptr ptrauth (ptr getelementptr (i8, ptr @g_weak, i64 16), i32 2, i64 42)
}

;--- err4.ll

; RUN: not --crash llc < err4.ll -mtriple aarch64-elf -mattr=+pauth \
; RUN:   -global-isel=1 -verify-machineinstrs -global-isel-abort=1 2>&1 | \
; RUN:   FileCheck --check-prefix=ERR4 %s

@g_weak = extern_weak global i32
@g_weak.ref.da.42.addr = dso_local constant ptr ptrauth (ptr @g_weak, i32 2, i64 42, ptr @g_weak.ref.da.42.addr)

define ptr @foo() {
; ERR4: LLVM ERROR: unsupported weak addr-div ptrauth global
  ret ptr ptrauth (ptr @g_weak, i32 0, i64 42, ptr @g_weak.ref.da.42.addr)
}

;--- err5.ll

; RUN: not --crash llc < err5.ll -mtriple arm64-apple-darwin -mattr=+pauth \
; RUN:   -global-isel=1 -verify-machineinstrs -global-isel-abort=1 2>&1 | \
; RUN:   FileCheck --check-prefix=ERR5 %s

@g = external global i32

define ptr @foo() {
; ERR5: LLVM ERROR: ptrauth global lowering is only implemented for ELF
  ret ptr ptrauth (ptr @g, i32 0)
}

;--- ok.ll

; RUN: llc < ok.ll -mtriple aarch64-elf -mattr=+pauth -global-isel=1 \
; RUN:   -verify-machineinstrs -global-isel-abort=1 | FileCheck %s
; RUN: llc < ok.ll -mtriple aarch64-elf -mattr=+pauth -global-isel=1 \
; RUN:   -verify-machineinstrs -global-isel-abort=1 -filetype=obj

@g = external global i32
@g_weak = extern_weak global i32
@g_strong_def = dso_local constant i32 42

define ptr @test_global_zero_disc() {
; CHECK-LABEL: test_global_zero_disc:
; CHECK:       // %bb.0:
; CHECK-NEXT:    adrp    x16, :got:g
; CHECK-NEXT:    ldr     x16, [x16, :got_lo12:g]
; CHECK-NEXT:    paciza  x16
; CHECK-NEXT:    mov     x0, x16
; CHECK-NEXT:    ret

  ret ptr ptrauth (ptr @g, i32 0)
}

define ptr @test_global_offset_zero_disc() {
; CHECK-LABEL: test_global_offset_zero_disc:
; CHECK:       // %bb.0:
; CHECK-NEXT:    adrp    x16, :got:g
; CHECK-NEXT:    ldr     x16, [x16, :got_lo12:g]
; CHECK-NEXT:    add     x16, x16, #16
; CHECK-NEXT:    pacdza  x16
; CHECK-NEXT:    mov     x0, x16
; CHECK-NEXT:    ret

  ret ptr ptrauth (ptr getelementptr (i8, ptr @g, i64 16), i32 2)
}

define ptr @test_global_neg_offset_zero_disc() {
; CHECK-LABEL: test_global_neg_offset_zero_disc:
; CHECK:       // %bb.0:
; CHECK-NEXT:    adrp    x16, :got:g
; CHECK-NEXT:    ldr     x16, [x16, :got_lo12:g]
; CHECK-NEXT:    sub     x16, x16, #576
; CHECK-NEXT:    sub     x16, x16, #30, lsl #12
; CHECK-NEXT:    pacdza  x16
; CHECK-NEXT:    mov     x0, x16
; CHECK-NEXT:    ret

  ret ptr ptrauth (ptr getelementptr (i8, ptr @g, i64 -123456), i32 2)
}

define ptr @test_global_big_offset_zero_disc() {
; CHECK-LABEL: test_global_big_offset_zero_disc:
; CHECK:       // %bb.0:
; CHECK-NEXT:    adrp    x16, :got:g
; CHECK-NEXT:    ldr     x16, [x16, :got_lo12:g]
; CHECK-NEXT:    mov     x17, #1
; CHECK-NEXT:    movk    x17, #32769, lsl #16
; CHECK-NEXT:    add     x16, x16, x17
; CHECK-NEXT:    pacdza  x16
; CHECK-NEXT:    mov     x0, x16
; CHECK-NEXT:    ret

  ret ptr ptrauth (ptr getelementptr (i8, ptr @g, i64 add (i64 2147483648, i64 65537)), i32 2)
}

define ptr @test_global_big_neg_offset_zero_disc() {
; CHECK-LABEL: test_global_big_neg_offset_zero_disc:
; CHECK:       // %bb.0:
; CHECK-NEXT:    adrp    x16, :got:g
; CHECK-NEXT:    ldr     x16, [x16, :got_lo12:g]
; CHECK-NEXT:    mov     x17, #-52501
; CHECK-NEXT:    movk    x17, #63652, lsl #16
; CHECK-NEXT:    add     x16, x16, x17
; CHECK-NEXT:    pacdza  x16
; CHECK-NEXT:    mov     x0, x16
; CHECK-NEXT:    ret

  ret ptr ptrauth (ptr getelementptr (i8, ptr @g, i64 -123456789), i32 2)
}

define ptr @test_global_huge_neg_offset_zero_disc() {
; CHECK-LABEL: test_global_huge_neg_offset_zero_disc:
; CHECK:       // %bb.0:
; CHECK-NEXT:    adrp    x16, :got:g
; CHECK-NEXT:    ldr     x16, [x16, :got_lo12:g]
; CHECK-NEXT:    mov     x17, #-65536
; CHECK-NEXT:    movk    x17, #0, lsl #16
; CHECK-NEXT:    movk    x17, #0, lsl #32
; CHECK-NEXT:    movk    x17, #32768, lsl #48
; CHECK-NEXT:    add     x16, x16, x17
; CHECK-NEXT:    pacdza  x16
; CHECK-NEXT:    mov     x0, x16
; CHECK-NEXT:    ret

  ret ptr ptrauth (ptr getelementptr (i8, ptr @g, i64 -9223372036854775808), i32 2)
}

define ptr @test_global_disc() {
; CHECK-LABEL: test_global_disc:
; CHECK:       // %bb.0:
; CHECK-NEXT:    adrp    x16, :got:g
; CHECK-NEXT:    ldr     x16, [x16, :got_lo12:g]
; CHECK-NEXT:    mov     x17, #42 // =0x2a
; CHECK-NEXT:    pacia   x16, x17
; CHECK-NEXT:    mov     x0, x16
; CHECK-NEXT:    ret

  ret ptr ptrauth (ptr @g, i32 0, i64 42)
}

@g.ref.da.42.addr = dso_local constant ptr ptrauth (ptr @g, i32 2, i64 42, ptr @g.ref.da.42.addr)

define ptr @test_global_addr_disc() {
; CHECK-LABEL: test_global_addr_disc:
; CHECK:       // %bb.0:
; CHECK-NEXT:    adrp x8, g.ref.da.42.addr
; CHECK-NEXT:    add x8, x8, :lo12:g.ref.da.42.addr
; CHECK-NEXT:    adrp x16, :got:g
; CHECK-NEXT:    ldr x16, [x16, :got_lo12:g]
; CHECK-NEXT:    mov x17, x8
; CHECK-NEXT:    movk x17, #42, lsl #48
; CHECK-NEXT:    pacda x16, x17
; CHECK-NEXT:    mov x0, x16
; CHECK-NEXT:    ret

  ret ptr ptrauth (ptr @g, i32 2, i64 42, ptr @g.ref.da.42.addr)
}

define ptr @test_global_process_specific() {
; CHECK-LABEL: test_global_process_specific:
; CHECK:       // %bb.0:
; CHECK-NEXT:    adrp    x16, :got:g
; CHECK-NEXT:    ldr     x16, [x16, :got_lo12:g]
; CHECK-NEXT:    pacizb  x16
; CHECK-NEXT:    mov     x0, x16
; CHECK-NEXT:    ret
  ret ptr ptrauth (ptr @g, i32 1)
}

; weak symbols can't be assumed to be non-nil. Use $auth_ptr$ stub slot always.
; The alternative is to emit a null-check here, but that'd be redundant with
; whatever null-check follows in user code.

define ptr @test_global_weak() {
; CHECK-LABEL: test_global_weak:
; CHECK:       // %bb.0:
; CHECK-NEXT:    adrp    x0, g_weak$auth_ptr$ia$42
; CHECK-NEXT:    ldr     x0, [x0, :lo12:g_weak$auth_ptr$ia$42]
; CHECK-NEXT:    ret
  ret ptr ptrauth (ptr @g_weak, i32 0, i64 42)
}

; Non-external symbols don't need to be accessed through the GOT.

define ptr @test_global_strong_def() {
; CHECK-LABEL: test_global_strong_def:
; CHECK:       // %bb.0:
; CHECK-NEXT:    adrp    x16, g_strong_def
; CHECK-NEXT:    add     x16, x16, :lo12:g_strong_def
; CHECK-NEXT:    pacdza  x16
; CHECK-NEXT:    mov     x0, x16
; CHECK-NEXT:    ret
  ret ptr ptrauth (ptr @g_strong_def, i32 2)
}

; CHECK-LABEL: g_weak$auth_ptr$ia$42:
; CHECK-NEXT:    .xword  g_weak@AUTH(ia,42)
