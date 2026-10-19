; RUN: llc -mtriple=x86_64-pc-win32 %s -o - | FileCheck %s --check-prefix=X64
; RUN: llc -mtriple=x86_64-pc-win32 -filetype=obj %s -o - | llvm-readobj -r - | FileCheck %s --check-prefix=RELOC
; RUN: llc -mtriple=x86_64-pc-win32 -filetype=obj %s -o - | llvm-readobj -S --section-data - | FileCheck %s --check-prefix=DATA

@__ImageBase = external global i8

; X64: .long   "?x@@3HA"@IMGREL
; RELOC: IMAGE_REL_AMD64_ADDR32NB ?x@@3HA
@"\01?x@@3HA" = global i32 trunc (i64 sub nsw (i64 ptrtoint (ptr @"\01?x@@3HA" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), align 4

declare dllimport void @f()

; X64: .long   f@IMGREL
; RELOC: IMAGE_REL_AMD64_ADDR32NB f
@fp = global i32 trunc (i64 sub nsw (i64 ptrtoint (ptr @f to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), align 4

@target = internal global i32 42
@alias = hidden alias i32, ptr @target

; X64: .long   alias@IMGREL
; RELOC: IMAGE_REL_AMD64_ADDR32NB alias
@alias_ref = global i32 trunc (i64 sub nsw (i64 ptrtoint (ptr @alias to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), align 4

define internal void @func() {
  ret void
}
@func_alias = hidden alias void (), ptr @func

; X64: .long   func_alias@IMGREL
; RELOC: IMAGE_REL_AMD64_ADDR32NB func_alias
@func_alias_ref = global i32 trunc (i64 sub nsw (i64 ptrtoint (ptr @func_alias to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), align 4

; X64: .long   alias@IMGREL+4
; RELOC: IMAGE_REL_AMD64_ADDR32NB alias
; DATA:      Name: .data
; DATA:      SectionData (
; DATA-NEXT:   0000: {{[0-9A-F ]+}}
; DATA-NEXT:   0010: 00000000 04000000 |
; DATA-NEXT: )
@alias_addend_ref = global i32 trunc (i64 sub nsw (i64 ptrtoint (ptr getelementptr (i32, ptr @alias, i32 1) to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), align 4
