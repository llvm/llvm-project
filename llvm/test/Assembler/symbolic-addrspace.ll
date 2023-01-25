;; Check that we can parse symbolic addres space constants "A", "G", "P".
;; NB: These do not round-trip via llvm-as, they are purely for initial parsing
;; and will be converted to a numerical constant that does not depend on the
;; datalayout by the .ll parser.
; RUN: split-file %s %t --leading-lines
; RUN: llvm-as < %t/valid.ll | llvm-dis | FileCheck %s
; RUN: llvm-as < %t/alloca-in-other-as.ll | llvm-dis | FileCheck %s --check-prefix ALLOCA-IN-GLOBALS
; RUN: not llvm-as < %t/bad-not-string.ll 2>&1 | FileCheck %s --check-prefix=ERR-NOT-STR
; RUN: not llvm-as < %t/bad-unknown-char.ll 2>&1 | FileCheck %s --check-prefix=ERR-BAD-CHAR
; RUN: not llvm-as < %t/bad-multiple-valid-chars.ll 2>&1 | FileCheck %s --check-prefix=ERR-MULTIPLE-CHARS
; RUN: not llvm-as < %t/bad-using-at-symbol.ll 2>&1 | FileCheck %s --check-prefix=ERR-AT-SYMBOL
; RUN: not llvm-as < %t/bad-number-in-quotes.ll 2>&1 | FileCheck %s --check-prefix=ERR-NUMBER-IN-QUOTES

;--- valid.ll
target datalayout = "A1-G2-P3"
; CHECK: target datalayout = "A1-G2-P3"

; CHECK: @str = private addrspace(2) constant [4 x i8] c"str\00"
@str = private addrspace("G") constant [4 x i8] c"str\00"

define void @foo() {
  ; CHECK: %alloca = alloca i32, align 4, addrspace(1)
  %alloca = alloca i32, addrspace("A")
  ret void
}

; CHECK: define void @bar() addrspace(3) {
define void @bar() addrspace("P") {
  ; CHECK: call addrspace(3) void @foo()
  call addrspace("P") void @foo()
  ret void
}

;--- alloca-in-other-as.ll
target datalayout = "A1-G2-P3"
; ALLOCA-IN-GLOBALS: target datalayout = "A1-G2-P3"

define void @foo() {
  ; ALLOCA-IN-GLOBALS: %alloca = alloca i32, align 4, addrspace(2){{$}}
  ; ALLOCA-IN-GLOBALS: %alloca2 = alloca i32, align 4, addrspace(1){{$}}
  ; ALLOCA-IN-GLOBALS: %alloca3 = alloca i32, align 4{{$}}
  ; ALLOCA-IN-GLOBALS: %alloca4 = alloca i32, align 4, addrspace(3){{$}}
  %alloca = alloca i32, addrspace("G")
  %alloca2 = alloca i32, addrspace("A")
  %alloca3 = alloca i32
  %alloca4 = alloca i32, addrspace("P")
  ret void
}

;--- bad-not-string.ll
target datalayout = "G2"
@str = private addrspace(D) constant [4 x i8] c"str\00"
; ERR-NOT-STR: [[#@LINE-1]]:26: error: expected integer or string constant

;--- bad-unknown-char.ll
target datalayout = "G2"
@str = private addrspace("D") constant [4 x i8] c"str\00"
; ERR-BAD-CHAR: [[#@LINE-1]]:26: error: invalid symbolic addrspace 'D'

;--- bad-multiple-valid-chars.ll
target datalayout = "A1-G2"
@str = private addrspace("AG") constant [4 x i8] c"str\00"
; ERR-MULTIPLE-CHARS: [[#@LINE-1]]:26: error: invalid symbolic addrspace 'AG'

;--- bad-using-at-symbol.ll
target datalayout = "A1-G2"
@str = private addrspace(@A) constant [4 x i8] c"str\00"
; ERR-AT-SYMBOL: [[#@LINE-1]]:26: error: expected integer or string constant

;--- bad-number-in-quotes.ll
target datalayout = "A1-G2"
@str = private addrspace("10") constant [4 x i8] c"str\00"
; ERR-NUMBER-IN-QUOTES: [[#@LINE-1]]:26: error: invalid symbolic addrspace '10'
