; RUN: llc -mtriple=arm64-apple-ios15 %s -o - | FileCheck %s --check-prefixes=CHECK-NOAUTH,CHECK
; RUN: llc -mtriple=arm64-apple-ios15 -mcpu=apple-a13 %s -o - | FileCheck %s --check-prefixes=CHECK-NOAUTH,CHECK
; RUN: llc -mtriple=arm64e-apple-ios15 %s -o - | FileCheck %s --check-prefixes=CHECK-AUTH,CHECK

; Important details in prologue:
;   * x22 is stored just below x29
;   * Enough stack space is allocated for everything
define swifttailcc void @simple(i8* swiftasync %ctx) "frame-pointer"="all" {
; CHECK-LABEL: simple:
; CHECK: orr x29, x29, #0x100000000000000
; CHECK: sub sp, sp, #32
; CHECK: stp x29, x30, [sp, #16]

; CHECK-NOAUTH-DAG: str x22, [sp, #8]
; CHECK-AUTH: add x16, sp, #8
; CHECK-AUTH: movk x16, #49946, lsl #48
; CHECK-AUTH: mov x17, x22
; CHECK-AUTH: pacdb x17, x16
; CHECK-AUTH: str x17, [sp, #8]

; CHECK-DAG: add x29, sp, #16
; CHECK: .cfi_def_cfa w29, 16
; CHECK: .cfi_offset w30, -8
; CHECK: .cfi_offset w29, -16

;[...]

; CHECK: ldp x29, x30, [sp, #16]
; CHECK: and x29, x29, #0xefffffffffffffff
; CHECK: add sp, sp, #32

  ret void
}

define swifttailcc void @more_csrs(i8* swiftasync %ctx) "frame-pointer"="all" {
; CHECK-LABEL: more_csrs:
; CHECK: orr x29, x29, #0x100000000000000
; CHECK: str x23, [sp, #-32]!
; CHECK: stp x29, x30, [sp, #16]

; CHECK-NOAUTH-DAG: str x22, [sp, #8]
; CHECK-AUTH: add x16, sp, #8
; CHECK-AUTH: movk x16, #49946, lsl #48
; CHECK-AUTH: mov x17, x22
; CHECK-AUTH: pacdb x17, x16
; CHECK-AUTH: str x17, [sp, #8]

; CHECK-DAG: add x29, sp, #16
; CHECK: .cfi_def_cfa w29, 16
; CHECK: .cfi_offset w30, -8
; CHECK: .cfi_offset w29, -16
; CHECK: .cfi_offset w23, -32

; [...]

; CHECK: ldp x29, x30, [sp, #16]
; CHECK: ldr x23, [sp], #32
; CHECK: and x29, x29, #0xefffffffffffffff
  call void asm sideeffect "", "~{x23}"()
  ret void
}

define swifttailcc void @locals(i8* swiftasync %ctx) "frame-pointer"="all" {
; CHECK-LABEL: locals:
; CHECK: orr x29, x29, #0x100000000000000
; CHECK: sub sp, sp, #64
; CHECK: stp x29, x30, [sp, #48]

; CHECK-NOAUTH-DAG: str x22, [sp, #40]
; CHECK-AUTH: add x16, sp, #40
; CHECK-AUTH: movk x16, #49946, lsl #48
; CHECK-AUTH: mov x17, x22
; CHECK-AUTH: pacdb x17, x16
; CHECK-AUTH: str x17, [sp, #40]

; CHECK-DAG: add x29, sp, #48
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

define swifttailcc void @use_input_context(i8* swiftasync %ctx, i8** %ptr) "frame-pointer"="all" {
; CHECK-LABEL: use_input_context:

; CHECK-NOAUTH: str x22, [sp
; CHECK-AUTH: mov x17, x22

; CHECK-NOT: x22
; CHECK: str x22, [x0]

  store i8* %ctx, i8** %ptr
  ret void
}

define swifttailcc i8** @context_in_func() "frame-pointer"="non-leaf" {
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

define swifttailcc void @write_frame_context(i8* swiftasync %ctx, i8* %newctx) "frame-pointer"="non-leaf" {
; CHECK-LABEL: write_frame_context:
; CHECK: sub x[[ADDR:[0-9]+]], x29, #8
; CHECK: str x0, [x[[ADDR]]]
  %ptr = call i8** @llvm.swift.async.context.addr()
  store i8* %newctx, i8** %ptr
  ret void
}

define swifttailcc void @simple_fp_elim(i8* swiftasync %ctx) "frame-pointer"="non-leaf" {
; CHECK-LABEL: simple_fp_elim:
; CHECK-NOT: orr x29, x29, #0x100000000000000

  ret void
}

define swifttailcc void @large_frame(i8* swiftasync %ctx) "frame-pointer"="all" {
; CHECK-LABEL: large_frame:
; CHECK: str x28, [sp, #-32]!
; CHECK: stp x29, x30, [sp, #16]
; CHECK-NOAUTH-DAG: str x22, [sp, #8]
; CHECK-DAG: add x29, sp, #16
; CHECK: sub sp, sp, #1024
; [...]
; CHECK: add sp, sp, #1024
; CHECK: ldp x29, x30, [sp, #16]
; CHECK: ldr x28, [sp], #32
; CHECK: ret
  %var = alloca i8, i32 1024
  ret void
}

; Important point is that there is just one 8-byte gap in the CSR region (right
; now just above d8) to realign the stack.
define swifttailcc void @two_unpaired_csrs(i8* swiftasync) "frame-pointer"="all" {
; CHECK-LABEL: two_unpaired_csrs:
; CHECK: str d8, [sp, #-48]!
; CHECK: str x19, [sp, #16]
; CHECK: stp x29, x30, [sp, #32]
; CHECK-NOAUTH-DAG: str x22, [sp, #24]
; CHECK-DAG: add x29, sp, #32

; CHECK: .cfi_def_cfa w29, 16
; CHECK: .cfi_offset w30, -8
; CHECK: .cfi_offset w29, -16
; CHECK: .cfi_offset w19, -32
; CHECK: .cfi_offset b8, -48

  call void asm "","~{x19},~{d8}"()
  call swifttailcc void @bar(i32* undef)
  ret void
}
declare swifttailcc void @bar(i32*)
declare i8** @llvm.swift.async.context.addr()
