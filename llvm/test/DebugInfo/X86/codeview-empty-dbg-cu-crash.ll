; RUN: llc -mtriple=x86_64-pc-windows-msvc < %s | FileCheck %s

; CHECK: .file	"<stdin>"
; CHECK-NEXT: .section	.debug$S,"dr"
; CHECK-NEXT: .p2align	2, 0x0
; CHECK-NEXT: .long	4                               # Debug section magic
; CHECK-NEXT: .long	241
; CHECK-NEXT: .long	.Ltmp1-.Ltmp0                   # Subsection size
; CHECK-NEXT: .Ltmp0:
; CHECK-NEXT: .short	.Ltmp3-.Ltmp2                   # Record length
; CHECK-NEXT: .Ltmp2:
; CHECK-NEXT: .short	4353                            # Record kind: S_OBJNAME
; CHECK-NEXT: .long	0                               # Signature
; CHECK-NEXT: .byte	0                               # Object name
; CHECK-NEXT: .p2align	2, 0x0
; CHECK-NEXT: .Ltmp3:
; CHECK-NEXT: .short	.Ltmp5-.Ltmp4                   # Record length
; CHECK-NEXT: .Ltmp4:
; CHECK-NEXT: .short	4412                            # Record kind: S_COMPILE3
; CHECK-NEXT: .long	3                               # Flags and language
; CHECK-NEXT: .short	208                             # CPUType
; CHECK-NEXT: .short	0                               # Frontend version
; CHECK-NEXT: .short	0
; CHECK-NEXT: .short	0
; CHECK-NEXT: .short	0
; CHECK-NEXT: .short	{{[0-9]+}}                      # Backend version
; CHECK-NEXT: .short	0
; CHECK-NEXT: .short	0
; CHECK-NEXT: .short	0
; CHECK-NEXT: .asciz	"0"                             # Null-terminated compiler version string
; CHECK-NEXT: .p2align	2, 0x0
; CHECK-NEXT: .Ltmp5:
; CHECK-NEXT: .Ltmp1:
; CHECK-NEXT: .p2align	2, 0x0

!llvm.dbg.cu = !{}
!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
