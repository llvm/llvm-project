; RUN: split-file %s %t
; RUN: cat %t/main.ll %t/a.ll > %t/a2.ll
; RUN: cat %t/main.ll %t/b.ll > %t/b2.ll
; RUN: cat %t/main.ll %t/c.ll > %t/c2.ll
; RUN: cat %t/main.ll %t/d.ll > %t/d2.ll
; RUN: cat %t/main.ll %t/e.ll > %t/e2.ll
; RUN: cat %t/main.ll %t/f.ll > %t/f2.ll
; RUN: cat %t/main.ll %t/g.ll > %t/g2.ll
; RUN: cat %t/main.ll %t/h.ll > %t/h2.ll
; RUN: cat %t/main.ll %t/i.ll > %t/i2.ll
; RUN: cat %t/main.ll %t/j.ll > %t/j2.ll
; RUN: llc %t/a2.ll -verify-machineinstrs -o - | \
; RUN: FileCheck --check-prefix=CHECK --check-prefix=CHECK-NO-OFFSET %s
; RUN: llc %t/b2.ll -verify-machineinstrs -o - | \
; RUN: FileCheck --check-prefix=CHECK --check-prefix=CHECK-POSITIVE-OFFSET %s
; RUN: llc %t/c2.ll -verify-machineinstrs -o - | \
; RUN: FileCheck --check-prefix=CHECK --check-prefix=CHECK-NEGATIVE-OFFSET %s
; RUN: llc %t/d2.ll -verify-machineinstrs -o - | \
; RUN: FileCheck --check-prefix=CHECK --check-prefix=CHECK-NPOT-OFFSET %s
; RUN: llc %t/e2.ll -verify-machineinstrs -o - | \
; RUN: FileCheck --check-prefix=CHECK --check-prefix=CHECK-NPOT-NEG-OFFSET %s
; RUN: llc %t/f2.ll -verify-machineinstrs -o - | \
; RUN: FileCheck --check-prefix=CHECK --check-prefix=CHECK-257-OFFSET %s
; RUN: llc %t/g2.ll -verify-machineinstrs -o - | \
; RUN: FileCheck --check-prefix=CHECK --check-prefix=CHECK-MINUS-257-OFFSET %s

; XFAIL
; RUN: not --crash llc %t/h2.ll -o - 2>&1 | \
; RUN: FileCheck --check-prefix=CHECK-BAD-OFFSET %s
; RUN: not --crash llc %t/i2.ll -o - 2>&1 | \
; RUN: FileCheck --check-prefix=CHECK-BAD-OFFSET %s
; RUN: not --crash llc %t/j2.ll -o - 2>&1 | \
; RUN: FileCheck --check-prefix=CHECK-BAD-OFFSET %s

;--- main.ll

target triple = "aarch64-unknown-linux-gnu"

; Verify that we `mrs` from `SP_EL0` twice, rather than load from
; __stack_chk_guard.
define dso_local void @foo(i64 %t) local_unnamed_addr #0 {
; CHECK-LABEL:   foo:
; CHECK:         // %bb.0: // %entry
; CHECK-NEXT:    stp x29, x30, [sp, #-16]! // 16-byte Folded Spill
; CHECK-NEXT:    .cfi_def_cfa_offset 16
; CHECK-NEXT:    mov x29, sp
; CHECK-NEXT:    .cfi_def_cfa w29, 16
; CHECK-NEXT:    .cfi_offset w30, -8
; CHECK-NEXT:    .cfi_offset w29, -16
; CHECK-NEXT:    .cfi_remember_state
; CHECK-NEXT:    sub sp, sp, #16
; CHECK-NEXT:    mrs x8, SP_EL0
; CHECK-NEXT:    lsl x9, x0, #2
; CHECK-NO-OFFSET:       ldr x8, [x8]
; CHECK-POSITIVE-OFFSET: ldr x8, [x8, #8]
; CHECK-NEGATIVE-OFFSET: ldur x8, [x8, #-8]
; CHECK-NPOT-OFFSET:     ldur x8, [x8, #1]
; CHECK-NPOT-NEG-OFFSET: ldur x8, [x8, #-1]
; CHECK-257-OFFSET:      add x8, x8, #257
; CHECK-MINUS-257-OFFSET:      sub x8, x8, #257
; CHECK-NEXT:    add x9, x9, #15
; CHECK-NEXT:    and x9, x9, #0xfffffffffffffff0
; CHECK-257-OFFSET-NEXT: ldr x8, [x8]
; CHECK-MINUS-257-OFFSET-NEXT: ldr x8, [x8]
; CHECK-NEXT:    stur x8, [x29, #-8]
; CHECK-NEXT:    mov x8, sp
; CHECK-NEXT:    sub x0, x8, x9
; CHECK-NEXT:    mov sp, x0
; CHECK-NEXT:    bl baz
; CHECK-NEXT:    mrs x8, SP_EL0
; CHECK-NO-OFFSET:       ldr x8, [x8]
; CHECK-POSITIVE-OFFSET: ldr x8, [x8, #8]
; CHECK-NEGATIVE-OFFSET: ldur x8, [x8, #-8]
; CHECK-NPOT-OFFSET:     ldur x8, [x8, #1]
; CHECK-NPOT-NEG-OFFSET: ldur x8, [x8, #-1]
; CHECK-257-OFFSET:      add x8, x8, #257
; CHECK-257-OFFSET-NEXT: ldr x8, [x8]
; CHECK-MINUS-257-OFFSET:      sub x8, x8, #257
; CHECK-MINUS-257-OFFSET-NEXT: ldr x8, [x8]
; CHECK-NEXT:    ldur x9, [x29, #-8]
; CHECK-NEXT:    cmp x8, x9
; CHECK-NEXT:    b.ne .LBB0_2
; CHECK-NEXT:  // %bb.1: // %entry
; CHECK-NEXT:    mov sp, x29
; CHECK-NEXT:    .cfi_def_cfa wsp, 16
; CHECK-NEXT:    ldp x29, x30, [sp], #16 // 16-byte Folded Reload
; CHECK-NEXT:   .cfi_def_cfa_offset 0
; CHECK-NEXT:   .cfi_restore w30
; CHECK-NEXT:   .cfi_restore w29
; CHECK-NEXT:    ret
; CHECK-NEXT:  .LBB0_2: // %entry
; CHECK-NEXT:   .cfi_restore_state
; CHECK-NEXT:    bl __stack_chk_fail
; CHECK-NOT: __stack_chk_guard
entry:
  %vla = alloca i32, i64 %t, align 4
  call void @baz(ptr nonnull %vla)
  ret void
}

declare void @baz(ptr)

; CHECK-BAD-OFFSET: LLVM ERROR: Unable to encode Stack Protector Guard Offset

attributes #0 = { sspstrong uwtable }
!llvm.module.flags = !{!1, !2, !3}

!1 = !{i32 2, !"stack-protector-guard", !"sysreg"}
!2 = !{i32 2, !"stack-protector-guard-reg", !"sp_el0"}

;--- a.ll
!3 = !{i32 2, !"stack-protector-guard-offset", i32 0}
;--- b.ll
!3 = !{i32 2, !"stack-protector-guard-offset", i32 8}
;--- c.ll
!3 = !{i32 2, !"stack-protector-guard-offset", i32 -8}
;--- d.ll
!3 = !{i32 2, !"stack-protector-guard-offset", i32 1}
;--- e.ll
!3 = !{i32 2, !"stack-protector-guard-offset", i32 -1}
;--- f.ll
!3 = !{i32 2, !"stack-protector-guard-offset", i32 257}
;--- g.ll
!3 = !{i32 2, !"stack-protector-guard-offset", i32 -257}
;--- h.ll
!3 = !{i32 2, !"stack-protector-guard-offset", i32 32761}
;--- i.ll
!3 = !{i32 2, !"stack-protector-guard-offset", i32 -4096}
;--- j.ll
!3 = !{i32 2, !"stack-protector-guard-offset", i32 4097}
