; RUN: rm -rf %t && split-file %s %t && cd %t

;--- nodisc.ll

; RUN: llc -mtriple aarch64-elf -mattr=+pauth -filetype=asm -o - nodisc.ll | \
; RUN:   FileCheck %s --check-prefix=ASM
; RUN: llc -mtriple aarch64-elf -mattr=+pauth -filetype=obj -o - nodisc.ll | \
; RUN:   llvm-readelf -r -x .init_array -x .fini_array - | FileCheck %s --check-prefix=OBJ

; ASM:      .section .init_array,"aw",@init_array
; ASM-NEXT: .p2align 3, 0x0
; ASM-NEXT: .xword   foo@AUTH(ia,55764)
; ASM-NEXT: .section .fini_array,"aw",@fini_array
; ASM-NEXT: .p2align 3, 0x0
; ASM-NEXT: .xword   bar@AUTH(ia,55764)

; OBJ:      Relocation section '.rela.init_array' at offset 0x[[#]] contains 1 entries:
; OBJ-NEXT:     Offset             Info             Type               Symbol's Value  Symbol's Name + Addend
; OBJ-NEXT: 0000000000000000  0000000700000244 R_AARCH64_AUTH_ABS64   0000000000000000 foo + 0
; OBJ:      Relocation section '.rela.fini_array' at offset 0x[[#]] contains 1 entries:
; OBJ-NEXT:     Offset             Info             Type               Symbol's Value  Symbol's Name + Addend
; OBJ-NEXT: 0000000000000000  0000000800000244 R_AARCH64_AUTH_ABS64   0000000000000004 bar + 0
; OBJ:      Hex dump of section '.init_array':
; OBJ-NEXT: 0x00000000 00000000 d4d90000
; OBJ:      Hex dump of section '.fini_array':
; OBJ-NEXT: 0x00000000 00000000 d4d90000
;;                              ^^^^ 0xD9D4: constant discriminator = 55764
;;                                    ^^ 0x80: bits 61..60 key = IA; bit 63 addr disc = false

@llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr ptrauth (ptr @foo, i32 0, i64 55764), ptr null }]
@llvm.global_dtors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr ptrauth (ptr @bar, i32 0, i64 55764), ptr null }]

define void @foo() {
  ret void
}

define void @bar() {
  ret void
}

;--- disc.ll

; RUN: llc -mtriple aarch64-elf -mattr=+pauth -filetype=asm -o - disc.ll | \
; RUN:   FileCheck %s --check-prefix=ASM-DISC
; RUN: llc -mtriple aarch64-elf -mattr=+pauth -filetype=obj -o - disc.ll | \
; RUN:   llvm-readelf -r -x .init_array -x .fini_array - | FileCheck %s --check-prefix=OBJ-DISC

; ASM-DISC:      .section .init_array,"aw",@init_array
; ASM-DISC-NEXT: .p2align 3, 0x0
; ASM-DISC-NEXT: .xword   foo@AUTH(ia,55764,addr)
; ASM-DISC-NEXT: .section .fini_array,"aw",@fini_array
; ASM-DISC-NEXT: .p2align 3, 0x0
; ASM-DISC-NEXT: .xword   bar@AUTH(ia,55764,addr)

; OBJ-DISC:      Relocation section '.rela.init_array' at offset 0x[[#]] contains 1 entries:
; OBJ-DISC-NEXT:     Offset             Info             Type               Symbol's Value  Symbol's Name + Addend
; OBJ-DISC-NEXT: 0000000000000000  0000000700000244 R_AARCH64_AUTH_ABS64   0000000000000000 foo + 0
; OBJ-DISC:      Relocation section '.rela.fini_array' at offset 0x[[#]] contains 1 entries:
; OBJ-DISC-NEXT:     Offset             Info             Type               Symbol's Value  Symbol's Name + Addend
; OBJ-DISC-NEXT: 0000000000000000  0000000800000244 R_AARCH64_AUTH_ABS64   0000000000000004 bar + 0
; OBJ-DISC:      Hex dump of section '.init_array':
; OBJ-DISC-NEXT: 0x00000000 00000000 d4d90080
; OBJ-DISC:      Hex dump of section '.fini_array':
; OBJ-DISC-NEXT: 0x00000000 00000000 d4d90080
;;                                   ^^^^ 0xD9D4: constant discriminator = 55764
;;                                         ^^ 0x80: bits 61..60 key = IA; bit 63 addr disc = true

@llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr ptrauth (ptr @foo, i32 0, i64 55764, ptr inttoptr (i64 1 to ptr)), ptr null }]
@llvm.global_dtors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr ptrauth (ptr @bar, i32 0, i64 55764, ptr inttoptr (i64 1 to ptr)), ptr null }]

define void @foo() {
  ret void
}

define void @bar() {
  ret void
}

;--- err1.ll

; RUN: not --crash llc -mtriple aarch64-elf -mattr=+pauth -filetype=asm -o - err1.ll 2>&1 | \
; RUN:   FileCheck %s --check-prefix=ERR1

; ERR1: LLVM ERROR: unexpected address discrimination value for ctors/dtors entry, only 'ptr inttoptr (i64 1 to ptr)' is allowed

@llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr ptrauth (ptr @foo, i32 0, i64 55764, ptr inttoptr (i64 2 to ptr)), ptr null }]

define void @foo() {
  ret void
}

;--- err2.ll

; RUN: not --crash llc -mtriple aarch64-elf -mattr=+pauth -filetype=asm -o - err2.ll 2>&1 | \
; RUN:   FileCheck %s --check-prefix=ERR2

; ERR2: LLVM ERROR: unexpected address discrimination value for ctors/dtors entry, only 'ptr inttoptr (i64 1 to ptr)' is allowed

@g = external global ptr
@llvm.global_dtors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr ptrauth (ptr @bar, i32 0, i64 55764, ptr @g), ptr null }]

define void @bar() {
  ret void
}
