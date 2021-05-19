; RUN: llc %s --stack-protector-guard=sysreg \
; RUN:   --stack-protector-guard-reg=sp_el0 \
; RUN:   --stack-protector-guard-offset=0 -verify-machineinstrs -o - | \
; RUN: FileCheck --check-prefix=CHECK --check-prefix=CHECK-NO-OFFSET %s
; RUN: llc %s --stack-protector-guard=sysreg \
; RUN:   --stack-protector-guard-reg=sp_el0 \
; RUN:   --stack-protector-guard-offset=8 -verify-machineinstrs -o - | \
; RUN: FileCheck --check-prefix=CHECK --check-prefix=CHECK-POSITIVE-OFFSET %s
; RUN: llc %s --stack-protector-guard=sysreg \
; RUN:   --stack-protector-guard-reg=sp_el0 \
; RUN:   --stack-protector-guard-offset=-8 -verify-machineinstrs -o - | \
; RUN: FileCheck --check-prefix=CHECK --check-prefix=CHECK-NEGATIVE-OFFSET %s
; RUN: llc %s --stack-protector-guard=sysreg \
; RUN:   --stack-protector-guard-reg=sp_el0 \
; RUN:   --stack-protector-guard-offset=1 -verify-machineinstrs -o - | \
; RUN: FileCheck --check-prefix=CHECK --check-prefix=CHECK-NPOT-OFFSET %s
; RUN: llc %s --stack-protector-guard=sysreg \
; RUN:   --stack-protector-guard-reg=sp_el0 \
; RUN:   --stack-protector-guard-offset=-1 -verify-machineinstrs -o - | \
; RUN: FileCheck --check-prefix=CHECK --check-prefix=CHECK-NPOT-NEG-OFFSET %s
; RUN: llc %s --stack-protector-guard=sysreg \
; RUN:   --stack-protector-guard-reg=sp_el0 \
; RUN:   --stack-protector-guard-offset=257 -verify-machineinstrs -o - | \
; RUN: FileCheck --check-prefix=CHECK --check-prefix=CHECK-257-OFFSET %s
; RUN: llc %s --stack-protector-guard=sysreg \
; RUN:   --stack-protector-guard-reg=sp_el0 \
; RUN:   --stack-protector-guard-offset=-257 -verify-machineinstrs -o - | \
; RUN: FileCheck --check-prefix=CHECK --check-prefix=CHECK-MINUS-257-OFFSET %s

; XFAIL
; RUN: not --crash llc %s --stack-protector-guard=sysreg \
; RUN:   --stack-protector-guard-reg=sp_el0 \
; RUN:   --stack-protector-guard-offset=32761 -o - 2>&1 | \
; RUN: FileCheck --check-prefix=CHECK-BAD-OFFSET %s
; RUN: not --crash llc %s --stack-protector-guard=sysreg \
; RUN:   --stack-protector-guard-reg=sp_el0 \
; RUN:   --stack-protector-guard-offset=-4096 -o - 2>&1 | \
; RUN: FileCheck --check-prefix=CHECK-BAD-OFFSET %s
; RUN: not --crash llc %s --stack-protector-guard=sysreg \
; RUN:   --stack-protector-guard-reg=sp_el0 \
; RUN:   --stack-protector-guard-offset=4097 -o - 2>&1 | \
; RUN: FileCheck --check-prefix=CHECK-BAD-OFFSET %s

target triple = "aarch64-unknown-linux-gnu"

; Verify that we `mrs` from `SP_EL0` twice, rather than load from
; __stack_chk_guard.
define dso_local void @foo(i64 %t) local_unnamed_addr #0 {
; CHECK-LABEL:   foo:
; CHECK:         // %bb.0: // %entry
; CHECK-NEXT:    stp x29, x30, [sp, #-16]! // 16-byte Folded Spill
; CHECK-NEXT:    mov x29, sp
; CHECK-NEXT:    sub sp, sp, #16 // =16
; CHECK-NEXT:    .cfi_def_cfa w29, 16
; CHECK-NEXT:    .cfi_offset w30, -8
; CHECK-NEXT:    .cfi_offset w29, -16
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
; CHECK-NEXT:    lsl x9, x0, #2
; CHECK-NEXT:    add x9, x9, #15 // =15
; CHECK-NEXT:    and x9, x9, #0xfffffffffffffff0
; CHECK-NEXT:    stur x8, [x29, #-8]
; CHECK-NEXT:    mov x8, sp
; CHECK-NEXT:    sub x0, x8, x9
; CHECK-NEXT:    mov sp, x0
; CHECK-NEXT:    bl baz
; CHECK-NEXT:    ldur x8, [x29, #-8]
; CHECK-NEXT:    mrs x9, SP_EL0
; CHECK-NO-OFFSET:       ldr x9, [x9]
; CHECK-POSITIVE-OFFSET: ldr x9, [x9, #8]
; CHECK-NEGATIVE-OFFSET: ldur x9, [x9, #-8]
; CHECK-NPOT-OFFSET:     ldur x9, [x9, #1]
; CHECK-NPOT-NEG-OFFSET: ldur x9, [x9, #-1]
; CHECK-257-OFFSET:      add x9, x9, #257
; CHECK-257-OFFSET-NEXT: ldr x9, [x9]
; CHECK-MINUS-257-OFFSET:      sub x9, x9, #257
; CHECK-MINUS-257-OFFSET-NEXT: ldr x9, [x9]
; CHECK-NEXT:    cmp x9, x8
; CHECK-NEXT:    b.ne .LBB0_2
; CHECK-NEXT:  // %bb.1: // %entry
; CHECK-NEXT:    mov sp, x29
; CHECK-NEXT:    ldp x29, x30, [sp], #16 // 16-byte Folded Reload
; CHECK-NEXT:    ret
; CHECK-NEXT:  .LBB0_2: // %entry
; CHECK-NEXT:    bl __stack_chk_fail
; CHECK-NOT: __stack_chk_guard
entry:
  %vla = alloca i32, i64 %t, align 4
  call void @baz(i32* nonnull %vla)
  ret void
}

declare void @baz(i32*)

attributes #0 = { sspstrong }

; CHECK-BAD-OFFSET: LLVM ERROR: Unable to encode Stack Protector Guard Offset
