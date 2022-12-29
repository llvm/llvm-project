; REQUIRES: bpf-registered-target

; RUN: llc -filetype obj -o - %s | llvm-readobj --sections - | FileCheck --check-prefix="SECTIONS" %s

; SECTIONS:         Name: .data.A
; SECTIONS-NEXT:    Type: SHT_PROGBITS (0x1)
; SECTIONS-NEXT:        Flags [ (0x3)
; SECTIONS-NEXT:          SHF_ALLOC (0x2)
; SECTIONS-NEXT:          SHF_WRITE (0x1)
; SECTIONS-NEXT:    ]
;
; SECTIONS:         Name: .rodata.A
; SECTIONS-NEXT:    Type: SHT_PROGBITS (0x1)
; SECTIONS-NEXT:        Flags [ (0x3)
; SECTIONS-NEXT:          SHF_ALLOC (0x2)
; SECTIONS-NEXT:          SHF_WRITE (0x1)
; SECTIONS-NEXT:    ]


target datalayout = "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128"
target triple = "bpf"

@glock = dso_local local_unnamed_addr global i32 0, section ".data.A", align 8
@ghead = dso_local local_unnamed_addr global i32 0, section ".data.A", align 8

@glock2 = dso_local local_unnamed_addr global i32 0, section ".rodata.A", align 8
@ghead2 = dso_local local_unnamed_addr global i32 0, section ".rodata.A", align 8
