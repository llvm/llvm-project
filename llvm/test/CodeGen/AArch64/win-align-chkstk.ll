; RUN: llc < %s -mtriple=aarch64-windows | FileCheck %s

define dso_local void @func() {
entry:
  %buf = alloca [8192 x i8], align 32
  %arraydecay = getelementptr inbounds [8192 x i8], ptr %buf, i64 0, i64 0
  call void @other(ptr noundef %arraydecay)
  ret void
}

declare dso_local void @other(ptr noundef)

; CHECK-LABEL: func:
; CHECK-NEXT: .seh_proc func
; CHECK-NEXT: // %bb.0:
; CHECK-NEXT: str x28, [sp, #-32]!
; CHECK-NEXT: .seh_save_reg_x x28, 32
; CHECK-NEXT: stp x29, x30, [sp, #8]
; CHECK-NEXT: .seh_save_fplr 8
; CHECK-NEXT: add x29, sp, #8
; CHECK-NEXT: .seh_add_fp 8
; CHECK-NEXT: .seh_endprologue
; CHECK-NEXT: mov x15, #513
; CHECK-NEXT: bl __chkstk
; CHECK-NEXT: sub sp, sp, x15, lsl #4
; CHECK-NEXT: add x15, sp, #16
; CHECK-NEXT: and sp, x15, #0xffffffffffffffe0
