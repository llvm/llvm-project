; REQUIRES: aarch64-registered-target
; RUN: llvm-mc -triple=arm64-apple-ios7.0 -filetype=obj %s -o %t.o

adrp x2, _var@TLVPPAGE
ldr x0, [x15, _var@TLVPPAGEOFF]
add x30, x0, _var@TLVPPAGEOFF

; RUN: llvm-objdump -rd %t.o | FileCheck %s --check-prefix=OBJDUMP
; RUN: llvm-objdump --macho -d --full-leading-addr --no-show-raw-insn %t.o \
; RUN:     | FileCheck %s --check-prefix=MACHO

; OBJDUMP:      adrp x2, 0x0
; OBJDUMP-NEXT:     0: ARM64_RELOC_TLVP_LOAD_PAGE21 _var
; OBJDUMP-NEXT: ldr x0, [x15]
; OBJDUMP-NEXT:     4: ARM64_RELOC_TLVP_LOAD_PAGEOFF12 _var
; OBJDUMP-NEXT: add x30, x0, #0
; OBJDUMP-NEXT:     8: ARM64_RELOC_TLVP_LOAD_PAGEOFF12 _var

; MACHO:      0000000000000000 adrp x2, _var@TLVPPAGE
; MACHO-NEXT: 0000000000000004 ldr x0, [x15, _var@TLVPPAGEOFF]
; MACHO-NEXT: 0000000000000008 add x30, x0, _var@TLVPPAGEOFF
