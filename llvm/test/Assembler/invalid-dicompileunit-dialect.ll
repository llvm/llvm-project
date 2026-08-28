; RUN: split-file %s %t
; RUN: not llvm-as < %t/string_literal.ll -disable-output 2>&1 | FileCheck %s --check-prefix=STRING_LITERAL
; RUN: not llvm-as < %t/invalid_enum.ll -disable-output 2>&1 | FileCheck %s --check-prefix=INVALID_ENUM
; RUN: not llvm-as < %t/wrong_enum_family.ll -disable-output 2>&1 | FileCheck %s --check-prefix=WRONG_ENUM
; RUN: not llvm-as < %t/zero.ll -disable-output 2>&1 | FileCheck %s --check-prefix=ZERO
; RUN: not llvm-as < %t/overflow.ll -disable-output 2>&1 | FileCheck %s --check-prefix=OVERFLOW

; STRING_LITERAL: expected DWARF language dialect
; INVALID_ENUM: invalid DWARF language dialect 'DW_LLVM_LANG_DIALECT_bogus'
; WRONG_ENUM: expected DWARF language dialect
;; Explicit numeric 0 is rejected: when the dialect field is present, it must
;; name one of the known dialects (simt or tile). To express "no dialect",
;; the field must be omitted entirely.
; ZERO: value for 'dialect' must be a known DWARF language dialect
;; The numeric dialect upper bound matches dwarf::DW_LLVM_LANG_DIALECT_max,
;; which is currently 2 (the highest defined dialect: DW_LLVM_LANG_DIALECT_tile).
; OVERFLOW: value for 'dialect' too large, limit is 2

;--- string_literal.ll
!0 = distinct !DICompileUnit(language: DW_LANG_C99, dialect: "simt",
                             file: !DIFile(filename: "a", directory: "b"))

;--- invalid_enum.ll
!0 = distinct !DICompileUnit(language: DW_LANG_C99,
                             dialect: DW_LLVM_LANG_DIALECT_bogus,
                             file: !DIFile(filename: "a", directory: "b"))

;--- wrong_enum_family.ll
!0 = distinct !DICompileUnit(language: DW_LANG_C99, dialect: DW_LANG_C99,
                             file: !DIFile(filename: "a", directory: "b"))

;--- zero.ll
!0 = distinct !DICompileUnit(language: DW_LANG_C99, dialect: 0,
                             file: !DIFile(filename: "a", directory: "b"))

;--- overflow.ll
!0 = distinct !DICompileUnit(language: DW_LANG_C99, dialect: 3,
                             file: !DIFile(filename: "a", directory: "b"))
