; RUN: rm -rf %t && split-file %s %t && cd %t

;--- const-1.ll

; RUN: llc -mtriple aarch64-elf -mattr=+pauth -filetype=asm -o - const-1.ll | \
; RUN:   FileCheck %s --check-prefix=ASM-CONST
; RUN: llc -mtriple aarch64-elf -mattr=+pauth -filetype=obj -o - const-1.ll | \
; RUN:   llvm-readelf -r -x .init_array -x .fini_array - | FileCheck %s --check-prefix=OBJ-CONST

; ASM-CONST:      .section .init_array,"aw",@init_array
; ASM-CONST-NEXT: .p2align 3, 0x0
; ASM-CONST-NEXT: .xword   foo@AUTH(ia,55764)
; ASM-CONST-NEXT: .section .fini_array,"aw",@fini_array
; ASM-CONST-NEXT: .p2align 3, 0x0
; ASM-CONST-NEXT: .xword   bar@AUTH(ia,55764)

; OBJ-CONST:      Relocation section '.rela.init_array' at offset 0x[[#]] contains 1 entries:
; OBJ-CONST-NEXT:     Offset             Info             Type               Symbol's Value  Symbol's Name + Addend
; OBJ-CONST-NEXT: 0000000000000000  0000000700000244 R_AARCH64_AUTH_ABS64   0000000000000000 foo + 0
; OBJ-CONST:      Relocation section '.rela.fini_array' at offset 0x[[#]] contains 1 entries:
; OBJ-CONST-NEXT:     Offset             Info             Type               Symbol's Value  Symbol's Name + Addend
; OBJ-CONST-NEXT: 0000000000000000  0000000800000244 R_AARCH64_AUTH_ABS64   0000000000000004 bar + 0
; OBJ-CONST:      Hex dump of section '.init_array':
; OBJ-CONST-NEXT: 0x00000000 00000000 d4d90000
; OBJ-CONST:      Hex dump of section '.fini_array':
; OBJ-CONST-NEXT: 0x00000000 00000000 d4d90000
;;                                    ^^^^ 0xD9D4: constant discriminator = 55764
;;                                          ^^ 0x80: bits 61..60 key = IA; bit 63 addr disc = false

@llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr @foo, ptr null }]
@llvm.global_dtors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr @bar, ptr null }]

define void @foo() {
  ret void
}

define void @bar() {
  ret void
}

!llvm.module.flags = !{!0, !1}
!0 = !{i32 1, !"ptrauth-init-fini", i32 1}
!1 = !{i32 1, !"ptrauth-init-fini-address-discrimination", i32 0}

;--- const-2.ll

; Use the same set of check lines as in const-1.ll, but check that absent flag
; is treated as having zero value.

; RUN: llc -mtriple aarch64-elf -mattr=+pauth -filetype=asm -o - const-2.ll | \
; RUN:   FileCheck %s --check-prefix=ASM-CONST
; RUN: llc -mtriple aarch64-elf -mattr=+pauth -filetype=obj -o - const-2.ll | \
; RUN:   llvm-readelf -r -x .init_array -x .fini_array - | FileCheck %s --check-prefix=OBJ-CONST

@llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr @foo, ptr null }]
@llvm.global_dtors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr @bar, ptr null }]

define void @foo() {
  ret void
}

define void @bar() {
  ret void
}

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"ptrauth-init-fini", i32 1}

;--- blended.ll

; RUN: llc -mtriple aarch64-elf -mattr=+pauth -filetype=asm -o - blended.ll | \
; RUN:   FileCheck %s --check-prefix=ASM-BLENDED
; RUN: llc -mtriple aarch64-elf -mattr=+pauth -filetype=obj -o - blended.ll | \
; RUN:   llvm-readelf -r -x .init_array -x .fini_array - | FileCheck %s --check-prefix=OBJ-BLENDED

; ASM-BLENDED:      .section .init_array,"aw",@init_array
; ASM-BLENDED-NEXT: .p2align 3, 0x0
; ASM-BLENDED-NEXT: .xword   foo@AUTH(ia,55764,addr)
; ASM-BLENDED-NEXT: .section .fini_array,"aw",@fini_array
; ASM-BLENDED-NEXT: .p2align 3, 0x0
; ASM-BLENDED-NEXT: .xword   bar@AUTH(ia,55764,addr)

; OBJ-BLENDED:      Relocation section '.rela.init_array' at offset 0x[[#]] contains 1 entries:
; OBJ-BLENDED-NEXT:     Offset             Info             Type               Symbol's Value  Symbol's Name + Addend
; OBJ-BLENDED-NEXT: 0000000000000000  0000000700000244 R_AARCH64_AUTH_ABS64   0000000000000000 foo + 0
; OBJ-BLENDED:      Relocation section '.rela.fini_array' at offset 0x[[#]] contains 1 entries:
; OBJ-BLENDED-NEXT:     Offset             Info             Type               Symbol's Value  Symbol's Name + Addend
; OBJ-BLENDED-NEXT: 0000000000000000  0000000800000244 R_AARCH64_AUTH_ABS64   0000000000000004 bar + 0
; OBJ-BLENDED:      Hex dump of section '.init_array':
; OBJ-BLENDED-NEXT: 0x00000000 00000000 d4d90080
; OBJ-BLENDED:      Hex dump of section '.fini_array':
; OBJ-BLENDED-NEXT: 0x00000000 00000000 d4d90080
;;                                      ^^^^ 0xD9D4: constant discriminator = 55764
;;                                            ^^ 0x80: bits 61..60 key = IA; bit 63 addr disc = true

@llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr @foo, ptr null }]
@llvm.global_dtors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr @bar, ptr null }]

define void @foo() {
  ret void
}

define void @bar() {
  ret void
}

!llvm.module.flags = !{!0, !1}
!0 = !{i32 1, !"ptrauth-init-fini", i32 1}
!1 = !{i32 1, !"ptrauth-init-fini-address-discrimination", i32 1}

;--- err1.ll

; RUN: not opt -S < err1.ll 2>&1 | FileCheck %s --check-prefix=ERR1

; ERR1: ptrauth-init-fini-address-discrimination module flag requires ptrauth-init-fini

@llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr @foo, ptr null }]

define void @foo() {
  ret void
}

!llvm.module.flags = !{!0, !1}
!0 = !{i32 1, !"ptrauth-init-fini", i32 0}
!1 = !{i32 1, !"ptrauth-init-fini-address-discrimination", i32 1}

;--- err2.ll

; RUN: not opt -S < err2.ll 2>&1 | FileCheck %s --check-prefix=ERR2

; ERR2: ptrauth-init-fini-address-discrimination module flag requires ptrauth-init-fini

@llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr @foo, ptr null }]

define void @foo() {
  ret void
}

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"ptrauth-init-fini-address-discrimination", i32 1}

;--- err3.ll

; RUN: not opt -S < err3.ll 2>&1 | FileCheck %s --check-prefix=ERR3

; ERR3: ptrauth-init-fini: module flag expects integer value

@llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr @foo, ptr null }]

define void @foo() {
  ret void
}

!llvm.module.flags = !{!0, !1}
!0 = !{i32 1, !"ptrauth-init-fini", !"1"}
!1 = !{i32 1, !"ptrauth-init-fini-address-discrimination", i32 1}

;--- err4.ll

; RUN: not opt -S < err4.ll 2>&1 | FileCheck %s --check-prefix=ERR4

; ERR4: ptrauth-init-fini-address-discrimination: module flag expects integer value

@llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr @foo, ptr null }]

define void @foo() {
  ret void
}

!llvm.module.flags = !{!0, !1}
!0 = !{i32 1, !"ptrauth-init-fini", i32 1}
!1 = !{i32 1, !"ptrauth-init-fini-address-discrimination", !"1"}
