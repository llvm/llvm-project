; RUN: split-file %s %t
; RUN: not llvm-as < %t/dw_lang_with_version.ll -disable-output 2>&1 | FileCheck %s --check-prefix=WRONG-ATTR
; RUN: not llvm-as < %t/overflow.ll -disable-output 2>&1 | FileCheck %s --check-prefix=OVERFLOW
; RUN: not llvm-as < %t/version_without_name.ll -disable-output 2>&1 | FileCheck %s --check-prefix=NO-NAME
; RUN: not llvm-as < %t/negative.ll -disable-output 2>&1 | FileCheck %s --check-prefix=NEGATIVE

; WRONG-ATTR: error: 'sourceLanguageVersion' requires an associated 'sourceLanguageName' on !DICompileUnit
; OVERFLOW: error: value for 'sourceLanguageVersion' too large, limit is 4294967295
; NEGATIVE: error: expected unsigned integer
; NO-NAME: error: missing one of 'language' or 'sourceLanguageName', required for !DICompileUnit

;--- dw_lang_with_version.ll
!0 = distinct !DICompileUnit(language: DW_LANG_C, sourceLanguageVersion: 1,
                             file: !DIFile(filename: "", directory: ""))

;--- overflow.ll
!0 = distinct !DICompileUnit(sourceLanguageName: DW_LNAME_C, sourceLanguageVersion: 4294967298)

;--- negative.ll
!0 = distinct !DICompileUnit(sourceLanguageName: DW_LNAME_C, sourceLanguageVersion: -1,
                             file: !DIFile(filename: "", directory: ""))

;--- version_without_name.ll
!0 = distinct !DICompileUnit(sourceLanguageVersion: 1,
                             file: !DIFile(filename: "", directory: ""))
