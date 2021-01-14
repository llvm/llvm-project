; RUN: llc -mtriple=arm64-apple-ios %s -o - | FileCheck %s --check-prefixes=CHECK-NOAUTH,CHECK
; RUN: llc -mtriple=arm64-apple-ios -mcpu=apple-a13 %s -o - | FileCheck %s --check-prefixes=CHECK-NOAUTH,CHECK
; RUN: llc -mtriple=arm64e-apple-ios %s -o - | FileCheck %s --check-prefixes=CHECK-AUTH,CHECK

; Important details in prologue:
;   * x22 is stored just below x29
;   * Enough stack space is allocated for everything
define void @simple(i8* swiftasync %ctx) "frame-pointer"="all" {
; CHECK-LABEL: simple:
; CHECK: orr x29, x29, #0x100000000000000
; CHECK: sub sp, sp, #32
; CHECK: stp x29, x30, [sp, #16]

; CHECK-NOAUTH: str x22, [sp, #8]
; CHECK-AUTH: add x16, sp, #8
; CHECK-AUTH: movk x16, #49946, lsl #48
; CHECK-AUTH: mov x17, x22
; CHECK-AUTH: pacdb x17, x16
; CHECK-AUTH: str x17, [sp, #8]

; CHECK: add x29, sp, #16
; CHECK: .cfi_def_cfa w29, 16
; CHECK: .cfi_offset w30, -8
; CHECK: .cfi_offset w29, -16

;[...]

; CHECK: ldp x29, x30, [sp, #16]
; CHECK: and x29, x29, #0xefffffffffffffff
; CHECK: add sp, sp, #32

  ret void
}

define void @more_csrs(i8* swiftasync %ctx) "frame-pointer"="all" {
; CHECK-LABEL: more_csrs:
; CHECK: orr x29, x29, #0x100000000000000
; CHECK: sub sp, sp, #48
; CHECK: stp x24, x23, [sp, #8]
; CHECK: stp x29, x30, [sp, #32]

; CHECK-NOAUTH: str x22, [sp, #24]
; CHECK-AUTH: add x16, sp, #24
; CHECK-AUTH: movk x16, #49946, lsl #48
; CHECK-AUTH: mov x17, x22
; CHECK-AUTH: pacdb x17, x16
; CHECK-AUTH: str x17, [sp, #24]

; CHECK: add x29, sp, #32
; CHECK: .cfi_def_cfa w29, 16
; CHECK: .cfi_offset w30, -8
; CHECK: .cfi_offset w29, -16
; CHECK: .cfi_offset w23, -32
; CHECK: .cfi_offset w24, -40

; [...]

; CHECK: ldp x29, x30, [sp, #32]
; CHECK: ldp x24, x23, [sp, #8]
; CHECK: and x29, x29, #0xefffffffffffffff
; CHECK: add sp, sp, #48
  call void asm sideeffect "", "~{x23}"()
  ret void
}

define void @locals(i8* swiftasync %ctx) "frame-pointer"="all" {
; CHECK-LABEL: locals:
; CHECK: orr x29, x29, #0x100000000000000
; CHECK: sub sp, sp, #64
; CHECK: stp x29, x30, [sp, #48]

; CHECK-NOAUTH: str x22, [sp, #40]
; CHECK-AUTH: add x16, sp, #40
; CHECK-AUTH: movk x16, #49946, lsl #48
; CHECK-AUTH: mov x17, x22
; CHECK-AUTH: pacdb x17, x16
; CHECK-AUTH: str x17, [sp, #40]

; CHECK: add x29, sp, #48
; CHECK: .cfi_def_cfa w29, 16
; CHECK: .cfi_offset w30, -8
; CHECK: .cfi_offset w29, -16

; CHECK: mov x0, sp
; CHECK: bl _bar

; [...]

; CHECK: ldp x29, x30, [sp, #48]
; CHECK: and x29, x29, #0xefffffffffffffff
; CHECK: add sp, sp, #64
  %var = alloca i32, i32 10
  call void @bar(i32* %var)
  ret void
}

define void @use_input_context(i8* swiftasync %ctx, i8** %ptr) "frame-pointer"="all" {
; CHECK-LABEL: use_input_context:

; CHECK-NOAUTH: str x22, [sp
; CHECK-AUTH: mov x17, x22

; CHECK-NOT: x22
; CHECK: str x22, [x0]

  store i8* %ctx, i8** %ptr
  ret void
}

define i8** @context_in_func() "frame-pointer"="non-leaf" {
; CHECK-LABEL: context_in_func:

; CHECK-NOAUTH: str xzr, [sp, #8]
; CHECK-AUTH: add x16, sp, #8
; CHECK-AUTH: movk x16, #49946, lsl #48
; CHECK-AUTH: mov x17, xzr
; CHECK-AUTH: pacdb x17, x16
; CHECK-AUTH: str x17, [sp, #8]

  %ptr = call i8** @llvm.swift.async.context.addr()
  ret i8** %ptr
}

define void @write_frame_context(i8* swiftasync %ctx, i8* %newctx) "frame-pointer"="non-leaf" {
; CHECK-LABEL: write_frame_context:
; CHECK: sub x[[ADDR:[0-9]+]], x29, #8
; CHECK: str x0, [x[[ADDR]]]
  %ptr = call i8** @llvm.swift.async.context.addr()
  store i8* %newctx, i8** %ptr
  ret void
}

define void @simple_fp_elim(i8* swiftasync %ctx) "frame-pointer"="non-leaf" {
; CHECK-LABEL: simple_fp_elim:
; CHECK-NOT: orr x29, x29, #0x100000000000000

  ret void
}

define void @large_frame(i8* swiftasync %ctx) "frame-pointer"="all" {
; CHECK-LABEL: large_frame:
; CHECK: sub sp, sp, #48
; CHECK: stp x28, x27, [sp, #8]
; CHECK: stp x29, x30, [sp, #32]
; CHECK-NOAUTH: str x22, [sp, #24]
; CHECK: add x29, sp, #32
; CHECK: sub sp, sp, #1024
; [...]
; CHECK: add sp, sp, #1024
; CHECK: ldp x29, x30, [sp, #32]
; CHECK: ldp x28, x27, [sp, #8]
; CHECK: add sp, sp, #48
; CHECK: ret
  %var = alloca i8, i32 1024
  ret void
}

declare void @bar(i32*)
declare i8** @llvm.swift.async.context.addr()
