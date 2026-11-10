; RUN: llc -mtriple=arm64-apple-macosx < %s | FileCheck %s --check-prefix=ASM
; RUN: llc -mtriple=arm64-apple-macosx -filetype=obj < %s | llvm-readobj --symbols - | FileCheck %s --check-prefix=OBJ

;; With subsections-via-symbols the linker splits a section into atoms at each
;; symbol, so the type id must not be a bare word before the function: it would
;; belong to the previous function's atom (here a cold one, which the linker
;; moves). Like prefix data, it gets its own label and the function becomes an
;; alternative entry point of it.

; ASM-LABEL: _f0:
; ASM:         .p2align 2
; ASM-NEXT:  {{ltmp[0-9]+}}:
; ASM-NEXT:    .long 12345678
; ASM-NEXT:    .alt_entry _f1
; ASM-NEXT:  _f1:

;; Prefix data already starts the function's atom; the type id follows it
;; under the same label.
; ASM:         .p2align 2
; ASM-NEXT:  {{ltmp[0-9]+}}:
; ASM-NEXT:    .long 1
; ASM-NEXT:    .alt_entry _f2
; ASM-NEXT:    .long 12345678
; ASM-NEXT:  _f2:

; OBJ-LABEL: Name: _f1
; OBJ:         Flags [ (0x200)
; OBJ-NEXT:      AltEntry (0x200)
; OBJ-NEXT:    ]
; OBJ-LABEL: Name: _f2
; OBJ:         Flags [ (0x200)
; OBJ-NEXT:      AltEntry (0x200)
; OBJ-NEXT:    ]

define void @f0() cold {
  ret void
}

define void @f1() !kcfi_type !0 {
  ret void
}

define void @f2() prefix i32 1 !kcfi_type !0 {
  ret void
}

!llvm.module.flags = !{!1}
!0 = !{i32 12345678}
!1 = !{i32 4, !"kcfi", i32 1}
