; RUN: rm -rf %t && split-file %s %t && cd %t

;--- err1.ll

; RUN: not --crash llc < err1.ll -mtriple aarch64-elf -mattr=+pauth \
; RUN:   -global-isel=1 -verify-machineinstrs -global-isel-abort=1 2>&1 | \
; RUN:   FileCheck --check-prefix=ERR1 %s
; RUN: not --crash llc < err1.ll -mtriple arm64-apple-ios -mattr=+pauth \
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
; RUN: not --crash llc < err2.ll -mtriple arm64-apple-ios -mattr=+pauth \
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
; RUN: not --crash llc < err3.ll -mtriple arm64-apple-ios -mattr=+pauth \
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
; RUN: not --crash llc < err4.ll -mtriple arm64-apple-ios -mattr=+pauth \
; RUN:   -global-isel=1 -verify-machineinstrs -global-isel-abort=1 2>&1 | \
; RUN:   FileCheck --check-prefix=ERR4 %s

@g_weak = extern_weak global i32
@g_weak.ref.da.42.addr = dso_local constant ptr ptrauth (ptr @g_weak, i32 2, i64 42, ptr @g_weak.ref.da.42.addr)

define ptr @foo() {
; ERR4: LLVM ERROR: unsupported weak addr-div ptrauth global
  ret ptr ptrauth (ptr @g_weak, i32 0, i64 42, ptr @g_weak.ref.da.42.addr)
}

;--- err5.ll

; RUN: not --crash llc < err5.ll -mtriple aarch64-windows -mattr=+pauth \
; RUN:   -global-isel=1 -verify-machineinstrs -global-isel-abort=1 2>&1 | \
; RUN:   FileCheck --check-prefix=ERR5 %s

@g = external global i32

define ptr @foo() {
; ERR5: LLVM ERROR: ptrauth global lowering only supported on MachO/ELF
  ret ptr ptrauth (ptr @g, i32 0)
}

;--- ok.ll

; RUN: llc < ok.ll -mtriple aarch64-elf -mattr=+pauth -global-isel=1 \
; RUN:   -verify-machineinstrs -global-isel-abort=1 | \
; RUN:   FileCheck %s --check-prefix=ELF
; RUN: llc < ok.ll -mtriple aarch64-elf -mattr=+pauth -global-isel=1 \
; RUN:   -verify-machineinstrs -global-isel-abort=1 -filetype=obj

; RUN: llc < ok.ll -mtriple arm64-apple-ios -mattr=+pauth -global-isel=1 \
; RUN:   -verify-machineinstrs -global-isel-abort=1 | \
; RUN:   FileCheck %s --check-prefix=MACHO
; RUN: llc < ok.ll -mtriple aarch64-elf -mattr=+pauth -global-isel=1 \
; RUN:   -verify-machineinstrs -global-isel-abort=1 -filetype=obj

@g = external global i32
@g_weak = extern_weak global i32
@g_strong_def = dso_local constant i32 42

define ptr @test_global_zero_disc() {
; ELF-LABEL:   test_global_zero_disc:
; ELF:         // %bb.0:
; ELF-NEXT:      adrp    x16, :got:g
; ELF-NEXT:      ldr     x16, [x16, :got_lo12:g]
; ELF-NEXT:      paciza  x16
; ELF-NEXT:      mov     x0, x16
; ELF-NEXT:      ret

; MACHO-LABEL: _test_global_zero_disc:
; MACHO:       ; %bb.0:
; MACHO-NEXT:    adrp    x16, _g@GOTPAGE
; MACHO-NEXT:    ldr     x16, [x16, _g@GOTPAGEOFF]
; MACHO-NEXT:    paciza  x16
; MACHO-NEXT:    mov     x0, x16
; MACHO-NEXT:    ret

  ret ptr ptrauth (ptr @g, i32 0)
}

define ptr @test_global_offset_zero_disc() {
; ELF-LABEL: test_global_offset_zero_disc:
; ELF:       // %bb.0:
; ELF-NEXT:    adrp    x16, :got:g
; ELF-NEXT:    ldr     x16, [x16, :got_lo12:g]
; ELF-NEXT:    add     x16, x16, #16
; ELF-NEXT:    pacdza  x16
; ELF-NEXT:    mov     x0, x16
; ELF-NEXT:    ret

; MACHO-LABEL: _test_global_offset_zero_disc:
; MACHO:       ; %bb.0:
; MACHO-NEXT:    adrp    x16, _g@GOTPAGE
; MACHO-NEXT:    ldr     x16, [x16, _g@GOTPAGEOFF]
; MACHO-NEXT:    add     x16, x16, #16
; MACHO-NEXT:    pacdza  x16
; MACHO-NEXT:    mov     x0, x16
; MACHO-NEXT:    ret

  ret ptr ptrauth (ptr getelementptr (i8, ptr @g, i64 16), i32 2)
}

define ptr @test_global_neg_offset_zero_disc() {
; ELF-LABEL: test_global_neg_offset_zero_disc:
; ELF:       // %bb.0:
; ELF-NEXT:    adrp    x16, :got:g
; ELF-NEXT:    ldr     x16, [x16, :got_lo12:g]
; ELF-NEXT:    sub     x16, x16, #576
; ELF-NEXT:    sub     x16, x16, #30, lsl #12
; ELF-NEXT:    pacdza  x16
; ELF-NEXT:    mov     x0, x16
; ELF-NEXT:    ret

; MACHO-LABEL: _test_global_neg_offset_zero_disc:
; MACHO:       ; %bb.0:
; MACHO-NEXT:    adrp    x16, _g@GOTPAGE
; MACHO-NEXT:    ldr     x16, [x16, _g@GOTPAGEOFF]
; MACHO-NEXT:    sub     x16, x16, #576
; MACHO-NEXT:    sub     x16, x16, #30, lsl #12
; MACHO-NEXT:    pacdza  x16
; MACHO-NEXT:    mov     x0, x16
; MACHO-NEXT:    ret

  ret ptr ptrauth (ptr getelementptr (i8, ptr @g, i64 -123456), i32 2)
}

define ptr @test_global_big_offset_zero_disc() {
; ELF-LABEL: test_global_big_offset_zero_disc:
; ELF:       // %bb.0:
; ELF-NEXT:    adrp    x16, :got:g
; ELF-NEXT:    ldr     x16, [x16, :got_lo12:g]
; ELF-NEXT:    mov     x17, #1
; ELF-NEXT:    movk    x17, #32769, lsl #16
; ELF-NEXT:    add     x16, x16, x17
; ELF-NEXT:    pacdza  x16
; ELF-NEXT:    mov     x0, x16
; ELF-NEXT:    ret

; MACHO-LABEL: _test_global_big_offset_zero_disc:
; MACHO:       ; %bb.0:
; MACHO-NEXT:    adrp    x16, _g@GOTPAGE
; MACHO-NEXT:    ldr     x16, [x16, _g@GOTPAGEOFF]
; MACHO-NEXT:    mov     x17, #1
; MACHO-NEXT:    movk    x17, #32769, lsl #16
; MACHO-NEXT:    add     x16, x16, x17
; MACHO-NEXT:    pacdza  x16
; MACHO-NEXT:    mov     x0, x16
; MACHO-NEXT:    ret

  ret ptr ptrauth (ptr getelementptr (i8, ptr @g, i64 add (i64 2147483648, i64 65537)), i32 2)
}

define ptr @test_global_big_neg_offset_zero_disc() {
; ELF-LABEL: test_global_big_neg_offset_zero_disc:
; ELF:       // %bb.0:
; ELF-NEXT:    adrp    x16, :got:g
; ELF-NEXT:    ldr     x16, [x16, :got_lo12:g]
; ELF-NEXT:    mov     x17, #-52501
; ELF-NEXT:    movk    x17, #63652, lsl #16
; ELF-NEXT:    add     x16, x16, x17
; ELF-NEXT:    pacdza  x16
; ELF-NEXT:    mov     x0, x16
; ELF-NEXT:    ret

; MACHO-LABEL: _test_global_big_neg_offset_zero_disc:
; MACHO:       ; %bb.0:
; MACHO-NEXT:    adrp    x16, _g@GOTPAGE
; MACHO-NEXT:    ldr     x16, [x16, _g@GOTPAGEOFF]
; MACHO-NEXT:    mov     x17, #-52501
; MACHO-NEXT:    movk    x17, #63652, lsl #16
; MACHO-NEXT:    add     x16, x16, x17
; MACHO-NEXT:    pacdza  x16
; MACHO-NEXT:    mov     x0, x16
; MACHO-NEXT:    ret

  ret ptr ptrauth (ptr getelementptr (i8, ptr @g, i64 -123456789), i32 2)
}

define ptr @test_global_huge_neg_offset_zero_disc() {
; ELF-LABEL: test_global_huge_neg_offset_zero_disc:
; ELF:       // %bb.0:
; ELF-NEXT:    adrp    x16, :got:g
; ELF-NEXT:    ldr     x16, [x16, :got_lo12:g]
; ELF-NEXT:    mov     x17, #-65536
; ELF-NEXT:    movk    x17, #0, lsl #16
; ELF-NEXT:    movk    x17, #0, lsl #32
; ELF-NEXT:    movk    x17, #32768, lsl #48
; ELF-NEXT:    add     x16, x16, x17
; ELF-NEXT:    pacdza  x16
; ELF-NEXT:    mov     x0, x16
; ELF-NEXT:    ret

; MACHO-LABEL: _test_global_huge_neg_offset_zero_disc:
; MACHO:       ; %bb.0:
; MACHO-NEXT:    adrp    x16, _g@GOTPAGE
; MACHO-NEXT:    ldr     x16, [x16, _g@GOTPAGEOFF]
; MACHO-NEXT:    mov     x17, #-65536
; MACHO-NEXT:    movk    x17, #0, lsl #16
; MACHO-NEXT:    movk    x17, #0, lsl #32
; MACHO-NEXT:    movk    x17, #32768, lsl #48
; MACHO-NEXT:    add     x16, x16, x17
; MACHO-NEXT:    pacdza  x16
; MACHO-NEXT:    mov     x0, x16
; MACHO-NEXT:    ret

  ret ptr ptrauth (ptr getelementptr (i8, ptr @g, i64 -9223372036854775808), i32 2)
}

define ptr @test_global_disc() {
; ELF-LABEL: test_global_disc:
; ELF:       // %bb.0:
; ELF-NEXT:    adrp    x16, :got:g
; ELF-NEXT:    ldr     x16, [x16, :got_lo12:g]
; ELF-NEXT:    mov     x17, #42 // =0x2a
; ELF-NEXT:    pacia   x16, x17
; ELF-NEXT:    mov     x0, x16
; ELF-NEXT:    ret

; MACHO-LABEL: _test_global_disc:
; MACHO:       ; %bb.0:
; MACHO-NEXT:    adrp    x16, _g@GOTPAGE
; MACHO-NEXT:    ldr     x16, [x16, _g@GOTPAGEOFF]
; MACHO-NEXT:    mov     x17, #42 ; =0x2a
; MACHO-NEXT:    pacia   x16, x17
; MACHO-NEXT:    mov     x0, x16
; MACHO-NEXT:    ret

  ret ptr ptrauth (ptr @g, i32 0, i64 42)
}

@g.ref.da.42.addr = dso_local constant ptr ptrauth (ptr @g, i32 2, i64 42, ptr @g.ref.da.42.addr)

define ptr @test_global_addr_disc() {
; ELF-LABEL: test_global_addr_disc:
; ELF:       // %bb.0:
; ELF-NEXT:    adrp x8, g.ref.da.42.addr
; ELF-NEXT:    add x8, x8, :lo12:g.ref.da.42.addr
; ELF-NEXT:    adrp x16, :got:g
; ELF-NEXT:    ldr x16, [x16, :got_lo12:g]
; ELF-NEXT:    mov x17, x8
; ELF-NEXT:    movk x17, #42, lsl #48
; ELF-NEXT:    pacda x16, x17
; ELF-NEXT:    mov x0, x16
; ELF-NEXT:    ret

; MACHO-LABEL: _test_global_addr_disc:
; MACHO:       ; %bb.0:
; MACHO-NEXT:   Lloh{{.*}}:
; MACHO-NEXT:    adrp x8, _g.ref.da.42.addr@PAGE
; MACHO-NEXT:   Lloh{{.*}}:
; MACHO-NEXT:    add x8, x8, _g.ref.da.42.addr@PAGEOFF
; MACHO-NEXT:    adrp x16, _g@GOTPAGE
; MACHO-NEXT:    ldr x16, [x16, _g@GOTPAGEOFF]
; MACHO-NEXT:    mov x17, x8
; MACHO-NEXT:    movk x17, #42, lsl #48
; MACHO-NEXT:    pacda x16, x17
; MACHO-NEXT:    mov x0, x16
; MACHO-NEXT:    ret

  ret ptr ptrauth (ptr @g, i32 2, i64 42, ptr @g.ref.da.42.addr)
}

define ptr @test_global_process_specific() {
; ELF-LABEL: test_global_process_specific:
; ELF:       // %bb.0:
; ELF-NEXT:    adrp    x16, :got:g
; ELF-NEXT:    ldr     x16, [x16, :got_lo12:g]
; ELF-NEXT:    pacizb  x16
; ELF-NEXT:    mov     x0, x16
; ELF-NEXT:    ret

; MACHO-LABEL: _test_global_process_specific:
; MACHO:       ; %bb.0:
; MACHO-NEXT:    adrp    x16, _g@GOTPAGE
; MACHO-NEXT:    ldr     x16, [x16, _g@GOTPAGEOFF]
; MACHO-NEXT:    pacizb  x16
; MACHO-NEXT:    mov     x0, x16
; MACHO-NEXT:    ret

  ret ptr ptrauth (ptr @g, i32 1)
}

; Non-external symbols don't need to be accessed through the GOT.

define ptr @test_global_strong_def() {
; ELF-LABEL: test_global_strong_def:
; ELF:       // %bb.0:
; ELF-NEXT:    adrp    x16, g_strong_def
; ELF-NEXT:    add     x16, x16, :lo12:g_strong_def
; ELF-NEXT:    pacdza  x16
; ELF-NEXT:    mov     x0, x16
; ELF-NEXT:    ret

; MACHO-LABEL: _test_global_strong_def:
; MACHO:       ; %bb.0:
; MACHO-NEXT:    adrp    x16, _g_strong_def@PAGE
; MACHO-NEXT:    add     x16, x16, _g_strong_def@PAGEOFF
; MACHO-NEXT:    pacdza  x16
; MACHO-NEXT:    mov     x0, x16
; MACHO-NEXT:    ret

  ret ptr ptrauth (ptr @g_strong_def, i32 2)
}

; weak symbols can't be assumed to be non-nil. Use $auth_ptr$ stub slot always.
; The alternative is to emit a null-check here, but that'd be redundant with
; whatever null-check follows in user code.

define ptr @test_global_weak() {
; ELF-LABEL: test_global_weak:
; ELF:       // %bb.0:
; ELF-NEXT:    adrp    x0, g_weak$auth_ptr$ia$42
; ELF-NEXT:    ldr     x0, [x0, :lo12:g_weak$auth_ptr$ia$42]
; ELF-NEXT:    ret

; MACHO-LABEL: _test_global_weak:
; MACHO:       ; %bb.0:
; MACHO-NEXT:    adrp    x0, l_g_weak$auth_ptr$ia$42@PAGE
; MACHO-NEXT:    ldr     x0, [x0, l_g_weak$auth_ptr$ia$42@PAGEOFF]
; MACHO-NEXT:    ret

  ret ptr ptrauth (ptr @g_weak, i32 0, i64 42)
}

; Test another weak symbol to check that stubs are emitted in a stable order.

@g_weak_2 = extern_weak global i32

define ptr @test_global_weak_2() {
; ELF-LABEL: test_global_weak_2:
; ELF:       // %bb.0:
; ELF-NEXT:    adrp    x0, g_weak_2$auth_ptr$ia$42
; ELF-NEXT:    ldr     x0, [x0, :lo12:g_weak_2$auth_ptr$ia$42]
; ELF-NEXT:    ret

; MACHO-LABEL: _test_global_weak_2:
; MACHO:       ; %bb.0:
; MACHO-NEXT:    adrp    x0, l_g_weak_2$auth_ptr$ia$42@PAGE
; MACHO-NEXT:    ldr     x0, [x0, l_g_weak_2$auth_ptr$ia$42@PAGEOFF]
; MACHO-NEXT:    ret

  ret ptr ptrauth (ptr @g_weak_2, i32 0, i64 42)
}

; ELF-LABEL: g_weak$auth_ptr$ia$42:
; ELF-NEXT:    .xword  g_weak@AUTH(ia,42)
; ELF-LABEL: g_weak_2$auth_ptr$ia$42:
; ELF-NEXT:    .xword  g_weak_2@AUTH(ia,42)

; MACHO-LABEL: l_g_weak$auth_ptr$ia$42:
; MACHO-NEXT:    .quad  _g_weak@AUTH(ia,42)
; MACHO-LABEL: l_g_weak_2$auth_ptr$ia$42:
; MACHO-NEXT:    .quad  _g_weak_2@AUTH(ia,42)
