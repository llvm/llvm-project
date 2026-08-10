;; RUN: llc -mtriple=aarch64-none-linux-gnu -filetype=obj %s -o - | \
;; RUN:   llvm-readobj -h -r - | FileCheck -check-prefix=OBJ %s

; Also take it on a round-trip through llvm-mc to stretch assembly-parsing's legs:
;; RUN: llc -mtriple=aarch64-none-linux-gnu %s -o - | \
;; RUN:     llvm-mc -triple=arm64-none-linux-gnu -filetype=obj -o - | \
;; RUN:     llvm-readobj -h -r - | FileCheck -check-prefix=OBJ %s

@var8 = dso_local global i8 0
@var16 = dso_local global i16 0
@var32 = dso_local global i32 0
@var64 = dso_local global i64 0

define dso_local void @loadstore() {
    %val8 = load i8, ptr @var8
    store volatile i8 %val8, ptr @var8

    %val16 = load i16, ptr @var16
    store volatile i16 %val16, ptr @var16

    %val32 = load i32, ptr @var32
    store volatile i32 %val32, ptr @var32

    %val64 = load i64, ptr @var64
    store volatile i64 %val64, ptr @var64

    ret void
}

@globaddr = dso_local global ptr null

define dso_local void @address() {
    store ptr @var64, ptr @globaddr
    ret void
}

; Check we're using EM_AARCH64
; OBJ: ElfHeader {
; OBJ:   Machine: EM_AARCH64
; OBJ: }

; OBJ: Relocations [
; OBJ:   Section {{.*}} .rela.text {
; OBJ:      0x{{[0-9,A-F]+}} R_AARCH64_ADR_PREL_PG_HI21 var8 0x0
; OBJ:      0x{{[0-9,A-F]+}} R_AARCH64_LDST8_ABS_LO12_NC var8 0x0
; OBJ:      0x{{[0-9,A-F]+}} R_AARCH64_LDST8_ABS_LO12_NC var8 0x0
; OBJ:      0x{{[0-9,A-F]+}} R_AARCH64_ADR_PREL_PG_HI21 var16 0x0
; OBJ:      0x{{[0-9,A-F]+}} R_AARCH64_LDST16_ABS_LO12_NC var16 0x0
; OBJ:      0x{{[0-9,A-F]+}} R_AARCH64_LDST16_ABS_LO12_NC var16 0x0
; OBJ:      0x{{[0-9,A-F]+}} R_AARCH64_ADR_PREL_PG_HI21 var32 0x0
; OBJ:      0x{{[0-9,A-F]+}} R_AARCH64_LDST32_ABS_LO12_NC var32 0x0
; OBJ:      0x{{[0-9,A-F]+}} R_AARCH64_LDST32_ABS_LO12_NC var32 0x0
; OBJ:      0x{{[0-9,A-F]+}} R_AARCH64_ADR_PREL_PG_HI21 var64 0x0
; OBJ:      0x{{[0-9,A-F]+}} R_AARCH64_LDST64_ABS_LO12_NC var64 0x0
; OBJ:      0x{{[0-9,A-F]+}} R_AARCH64_LDST64_ABS_LO12_NC var64 0x0
; OBJ:      0x{{[0-9,A-F]+}} R_AARCH64_ADR_PREL_PG_HI21 globaddr 0x0
; OBJ:      0x{{[0-9,A-F]+}} R_AARCH64_ADR_PREL_PG_HI21 var64 0x0
; OBJ:      0x{{[0-9,A-F]+}} R_AARCH64_ADD_ABS_LO12_NC var64 0x0
; OBJ:      0x{{[0-9,A-F]+}} R_AARCH64_LDST64_ABS_LO12_NC globaddr 0x0

; OBJ:   }
; OBJ: ]
