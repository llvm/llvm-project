; RUN: llc < %s -mtriple=aarch64-windows | FileCheck %s

define dso_local void @func() {
entry:
  %buf = alloca [64 x i8], align 32
  %arraydecay = getelementptr inbounds [64 x i8], ptr %buf, i64 0, i64 0
  call void @other(ptr noundef %arraydecay)
  ret void
}

declare dso_local void @other(ptr noundef)

; CHECK-LABEL: func:
; CHECK-NEXT: .seh_proc func
; CHECK-NEXT: // %bb.0:
; CHECK-NEXT: stp x29, x30, [sp, #-16]!
; CHECK-NEXT: .seh_save_fplr_x 16
; CHECK-NEXT: mov x29, sp
; CHECK-NEXT: .seh_set_fp
; CHECK-NEXT: sub x9, sp, #80
; CHECK-NEXT: .seh_nop
; CHECK-NEXT: and sp, x9, #0xffffffffffffffe0
; CHECK-NEXT: .seh_stackalloc 80
; CHECK-NEXT: .seh_endprologue

; FIXME: Ideally, the SEH prologue wouldn't include the stack realigning
; instructions at all, but it's enough to end the prologue after .seh_set_fp.
