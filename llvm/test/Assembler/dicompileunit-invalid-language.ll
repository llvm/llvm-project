; RUN: split-file %s %t
; RUN: not llvm-as < %t/invalid_dw_lang.ll -disable-output 2>&1 | FileCheck %s --check-prefix=INVALID_DW_LANG
; RUN: not llvm-as < %t/invalid_dw_lang_2.ll -disable-output 2>&1 | FileCheck %s --check-prefix=INVALID_DW_LANG_2
; RUN: not llvm-as < %t/invalid_dw_lname.ll -disable-output 2>&1 | FileCheck %s --check-prefix=INVALID_DW_LNAME
; RUN: not llvm-as < %t/invalid_dw_lname_2.ll -disable-output 2>&1 | FileCheck %s --check-prefix=INVALID_DW_LNAME_2

; INVALID_DW_LANG:    invalid DWARF language 'DW_LANG_blah'
; INVALID_DW_LANG_2:  expected DWARF language
; INVALID_DW_LNAME:   invalid DWARF source language name 'DW_LNAME_blah'
; INVALID_DW_LNAME_2: expected DWARF source language name

;--- invalid_dw_lang.ll
!0 = distinct !DICompileUnit(language: DW_LANG_blah)

;--- invalid_dw_lang_2.ll
!0 = distinct !DICompileUnit(language: DW_LNAME_C)

;--- invalid_dw_lname.ll
!0 = distinct !DICompileUnit(sourceLanguageName: DW_LNAME_blah)

;--- invalid_dw_lname_2.ll
!0 = distinct !DICompileUnit(sourceLanguageName: DW_LANG_C)
