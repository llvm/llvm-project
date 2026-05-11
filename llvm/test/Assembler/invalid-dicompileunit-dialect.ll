; RUN: split-file %s %t
; RUN: not llvm-as < %t/string_literal.ll -disable-output 2>&1 | FileCheck %s --check-prefix=STRING_LITERAL
; RUN: not llvm-as < %t/invalid_enum.ll -disable-output 2>&1 | FileCheck %s --check-prefix=INVALID_ENUM
; RUN: not llvm-as < %t/wrong_enum_family.ll -disable-output 2>&1 | FileCheck %s --check-prefix=WRONG_ENUM
; RUN: not llvm-as < %t/overflow.ll -disable-output 2>&1 | FileCheck %s --check-prefix=OVERFLOW

; STRING_LITERAL: expected DWARF language dialect
; INVALID_ENUM: invalid DWARF language dialect 'DW_LANG_DIALECT_bogus'
; WRONG_ENUM: expected DWARF language dialect
; OVERFLOW: value for 'dialect' too large, limit is 65535

;--- string_literal.ll
!0 = distinct !DICompileUnit(language: DW_LANG_C99, dialect: "simt",
                             file: !DIFile(filename: "a", directory: "b"))

;--- invalid_enum.ll
!0 = distinct !DICompileUnit(language: DW_LANG_C99,
                             dialect: DW_LANG_DIALECT_bogus,
                             file: !DIFile(filename: "a", directory: "b"))

;--- wrong_enum_family.ll
!0 = distinct !DICompileUnit(language: DW_LANG_C99, dialect: DW_LANG_C99,
                             file: !DIFile(filename: "a", directory: "b"))

;--- overflow.ll
!0 = distinct !DICompileUnit(language: DW_LANG_C99, dialect: 65536,
                             file: !DIFile(filename: "a", directory: "b"))
