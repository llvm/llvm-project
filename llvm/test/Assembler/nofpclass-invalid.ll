; RUN: rm -rf %t && split-file %s %t

; RUN: not llvm-as %t/nofpclass_0.ll -o /dev/null 2>&1 | FileCheck -check-prefix=MASKVALUE0 %s
; RUN: not llvm-as %t/nofpclass_1024.ll -o /dev/null 2>&1 | FileCheck -check-prefix=MASKVALUE1024 %s
; RUN: not llvm-as %t/nofpclass_two_numbers.ll -o /dev/null 2>&1 | FileCheck -check-prefix=TWONUMBERS %s
; RUN: not llvm-as %t/nofpclass_two_numbers_bar.ll -o /dev/null 2>&1 | FileCheck -check-prefix=TWONUMBERSBAR %s
; RUN: not llvm-as %t/nofpclass_two_numbers_neg1.ll -o /dev/null 2>&1 | FileCheck -check-prefix=MASKVALUENEG1 %s
; RUN: not llvm-as %t/nofpclass_only_keyword.ll -o /dev/null 2>&1 | FileCheck -check-prefix=ONLYKEYWORD %s
; RUN: not llvm-as %t/nofpclass_openparen.ll -o /dev/null 2>&1 | FileCheck -check-prefix=OPENPAREN %s
; RUN: not llvm-as %t/nofpclass_closeparen.ll -o /dev/null 2>&1 | FileCheck -check-prefix=CLOSEPAREN %s
; RUN: not llvm-as %t/nofpclass_emptyparens.ll -o /dev/null 2>&1 | FileCheck -check-prefix=EMPTYPARENS %s
; RUN: not llvm-as %t/nofpclass_0_missingparen.ll -o /dev/null 2>&1 | FileCheck -check-prefix=MISSINGPAREN0 %s
; RUN: not llvm-as %t/nofpclass_0_noparens.ll -o /dev/null 2>&1 | FileCheck -check-prefix=NOPARENS0 %s
; RUN: not llvm-as %t/nofpclass_1024_missing_paren.ll -o /dev/null 2>&1 | FileCheck -check-prefix=MISSINGPAREN1024 %s
; RUN: not llvm-as %t/nofpclass_neg1_missing_paren.ll -o /dev/null 2>&1 | FileCheck -check-prefix=MISSINGPAREN-NEGONE %s
; RUN: not llvm-as %t/nofpclass_1_noparens.ll -o /dev/null 2>&1 | FileCheck -check-prefix=NOPARENS-ONE %s
; RUN: not llvm-as %t/nofpclass_nan_noparens.ll -o /dev/null 2>&1 | FileCheck -check-prefix=NOPARENS-NAN %s
; RUN: not llvm-as %t/nofpclass_nnan_noparens.ll -o /dev/null 2>&1 | FileCheck -check-prefix=NOPARENS-NNAN %s
; RUN: not llvm-as %t/nofpclass_name_plus_int.ll -o /dev/null 2>&1 | FileCheck -check-prefix=NAME-PLUS-INT %s
; RUN: not llvm-as %t/nofpclass_name_follows_int.ll -o /dev/null 2>&1 | FileCheck -check-prefix=NAME-FOLLOWS-INT %s

;--- nofpclass_0.ll

; MASKVALUE0: error: invalid mask value for 'nofpclass'
define void @nofpclass_0(float nofpclass(0) %x) {
  ret void
}

;--- nofpclass_1024.ll

; MASKVALUE1024: error: invalid mask value for 'nofpclass'
define void @nofpclass_1024(float nofpclass(1024) %x) {
  ret void
}

;--- nofpclass_two_numbers.ll
; TWONUMBERS: error: expected ')'
define void @nofpclass_two_numbers(float nofpclass(2 4) %x) {
  ret void
}

;--- nofpclass_two_numbers_bar.ll
; TWONUMBERSBAR: error: expected ')'
define void @nofpclass_two_numbers_bar(float nofpclass(2|4) %x) {
  ret void
}

;--- nofpclass_two_numbers_neg1.ll
; MASKVALUENEG1: error: expected nofpclass test mask
define void @nofpclass_neg1(float nofpclass(-1) %x) {
  ret void
}

;--- nofpclass_only_keyword.ll
; ONLYKEYWORD: error: expected '('
define void @nofpclass_only_keyword(float nofpclass %x) {
  ret void
}

; FIXME: Poor diagnostic
;--- nofpclass_openparen.ll
; OPENPAREN: error: expected nofpclass test mask
define void @nofpclass_openparen(float nofpclass( %x) {
  ret void
}

;--- nofpclass_closeparen.ll
; CLOSEPAREN: error: expected '('
define void @nofpclass_closeparen(float nofpclass) %x) {
  ret void
}

;--- nofpclass_emptyparens.ll
; EMPTYPARENS: error: expected nofpclass test mask
define void @nofpclass_emptyparens(float nofpclass() %x) {
  ret void
}

; FIXME: Wrong error?
;--- nofpclass_0_missingparen.ll
; MISSINGPAREN0: error: invalid mask value for 'nofpclass'
define void @nofpclass_0_missingparen(float nofpclass(0 %x) {
  ret void
}

;--- nofpclass_0_noparens.ll
; NOPARENS0: error: expected '('
define void @nofpclass_0_noparens(float nofpclass 0 %x) {
  ret void
}

; FIXME: Wrong error
;--- nofpclass_1024_missing_paren.ll
; MISSINGPAREN1024: error: invalid mask value for 'nofpclass'
define void @nofpclass_1024_missing_paren(float nofpclass(1024 %x) {
  ret void
}

;--- nofpclass_neg1_missing_paren.ll
; MISSINGPAREN-NEGONE: error: expected nofpclass test mask
define void @nofpclass_neg1_missing_paren(float nofpclass(-1 %x) {
  ret void
}

;--- nofpclass_1_noparens.ll
; NOPARENS-ONE: error: expected '('
define void @nofpclass_1_noparens(float nofpclass 1 %x) {
  ret void
}

;--- nofpclass_nan_noparens.ll
; NOPARENS-NAN: error: expected '('
define void @nofpclass_nan_noparens(float nofpclass nan %x) {
  ret void
}

;--- nofpclass_nnan_noparens.ll
; NOPARENS-NNAN: error: expected '('
define void @nofpclass_nnan_noparens(float nofpclass nnan %x) {
  ret void
}

;--- nofpclass_name_plus_int.ll
; NAME-PLUS-INT: error: expected nofpclass test mask
define void @nofpclass_name_plus_int(float nofpclass(nan 42) %x) {
  ret void
}

;--- nofpclass_name_follows_int.ll
; NAME-FOLLOWS-INT: error: expected ')'
define void @nofpclass_name_plus_int(float nofpclass(42 nan) %x) {
  ret void
}
