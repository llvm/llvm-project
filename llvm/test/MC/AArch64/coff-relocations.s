; RUN: llvm-mc -triple aarch64-windows -filetype obj -o - %s | \
; RUN: llvm-readobj -r - | FileCheck %s

; IMAGE_REL_ARM64_ADDR32
.Linfo_foo:
  .asciz "foo"
  .long foo

; IMAGE_REL_ARM64_ADDR32NB
.long func@IMGREL

; IMAGE_REL_ARM64_ADDR64
.globl struc
struc:
  .quad arr

; IMAGE_REL_ARM64_BRANCH26
b target

; IMAGE_REL_ARM64_PAGEBASE_REL21
adrp x0, foo

; IMAGE_REL_ARM64_PAGEOFFSET_12A
add x0, x0, :lo12:foo

; IMAGE_REL_ARM64_PAGEOFFSET_12L
ldr x0, [x0, :lo12:foo]

; IMAGE_REL_ARM64_SECREL
.secrel32 .Linfo_bar
.Linfo_bar:

; IMAGE_REL_ARM64_SECTION
.secidx func


; CHECK: Format: COFF-ARM64
; CHECK: Arch: aarch64
; CHECK: AddressSize: 64bit
; CHECK: Relocations [
; CHECK:   Section (1) .text {
; CHECK: 0x4 IMAGE_REL_ARM64_ADDR32 foo
; CHECK: 0x8 IMAGE_REL_ARM64_ADDR32NB func
; CHECK: 0xC IMAGE_REL_ARM64_ADDR64 arr
; CHECK: 0x14 IMAGE_REL_ARM64_BRANCH26 target
; CHECK: 0x18 IMAGE_REL_ARM64_PAGEBASE_REL21 foo
; CHECK: 0x1C IMAGE_REL_ARM64_PAGEOFFSET_12A foo
; CHECK: 0x20 IMAGE_REL_ARM64_PAGEOFFSET_12L foo
; CHECK: 0x24 IMAGE_REL_ARM64_SECREL .text
; CHECK: 0x28 IMAGE_REL_ARM64_SECTION func
; CHECK:   }
; CHECK: ]
