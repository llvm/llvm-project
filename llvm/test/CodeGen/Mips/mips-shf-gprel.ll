; Check that .sdata section has SHF_MIPS_GPREL flag.

; RUN: llc -mips-ssection-threshold=16 -mgpopt -mattr=noabicalls \
; RUN:     -relocation-model=static -march=mips -o - %s -filetype=obj \
; RUN:   | llvm-readobj -s | FileCheck %s

@data1 = global [4 x i32] [i32 1, i32 2, i32 3, i32 4], align 4
@date2 = global [4 x i32] zeroinitializer, align 4

; CHECK:      Name: .sdata
; CHECK-NEXT: Type: SHT_PROGBITS
; CHECK-NEXT: Flags [ (0x10000003)
; CHECK-NEXT:   SHF_ALLOC
; CHECK-NEXT:   SHF_MIPS_GPREL
; CHECK-NEXT:   SHF_WRITE
; CHECK-NEXT: ]

; CHECK:      Name: .sbss
; CHECK-NEXT: Type: SHT_NOBITS
; CHECK-NEXT: Flags [ (0x10000003)
; CHECK-NEXT:   SHF_ALLOC
; CHECK-NEXT:   SHF_MIPS_GPREL
; CHECK-NEXT:   SHF_WRITE
; CHECK-NEXT: ]
