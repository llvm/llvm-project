; RUN: split-file --leading-lines %s %t
; RUN: not llvm-as < %t/missing-lparen.ll 2>&1 | FileCheck %s --check-prefix=CHECK-MISSING-LPAREN
; RUN: not llvm-as < %t/missing-rparen.ll 2>&1 | FileCheck %s --check-prefix=CHECK-MISSING-RPAREN
; RUN: not llvm-as < %t/missing-rparen-none.ll 2>&1 | FileCheck %s --check-prefix=CHECK-MISSING-RPAREN-NONE
; RUN: not llvm-as < %t/missing-colon.ll 2>&1 | FileCheck %s --check-prefix=CHECK-MISSING-COLON
; RUN: not llvm-as < %t/invalid-component.ll 2>&1 | FileCheck %s --check-prefix=CHECK-INVALID-COMPONENT
; RUN: not llvm-as < %t/duplicate-ret.ll 2>&1 | FileCheck %s --check-prefix=CHECK-DUPLICATE-RET
; RUN: not llvm-as < %t/none-after.ll 2>&1 | FileCheck %s --check-prefix=CHECK-NONE-AFTER
; RUN: not llvm-as < %t/none-before.ll 2>&1 | FileCheck %s --check-prefix=CHECK-NONE-BEFORE
; RUN: not opt -disable-output < %t/non-pointer-type.ll 2>&1 | FileCheck %s --check-prefix=CHECK-NON-POINTER-TYPE

;--- missing-lparen.ll

; CHECK-MISSING-LPAREN: <stdin>:[[@LINE+1]]:32: error: expected '('
define void @test(ptr captures %p) {
  ret void
}

;--- missing-rparen.ll

; CHECK-MISSING-RPAREN: <stdin>:[[@LINE+1]]:40: error: expected ',' or ')'
define void @test(ptr captures(address %p) {
  ret void
}

;--- missing-rparen-none.ll

; CHECK-MISSING-RPAREN-NONE: <stdin>:[[@LINE+1]]:37: error: expected ',' or ')'
define void @test(ptr captures(none %p) {
  ret void
}

;--- missing-colon.ll

; CHECK-MISSING-COLON: <stdin>:[[@LINE+1]]:36: error: expected ':'
define void @test(ptr captures(ret address) %p) {
  ret void
}

;--- invalid-component.ll

; CHECK-INVALID-COMPONENT: <stdin>:[[@LINE+1]]:32: error: expected one of 'none', 'address', 'address_is_null', 'provenance' or 'read_provenance'
define void @test(ptr captures(foo) %p) {
  ret void
}

;--- duplicate-ret.ll

; CHECK-DUPLICATE-RET: <stdin>:[[@LINE+1]]:51: error: duplicate 'ret' location
define void @test(ptr captures(ret: address, ret: provenance) %p) {
  ret void
}

;--- none-after.ll

; CHECK-NONE-AFTER: <stdin>:[[@LINE+1]]:45: error: cannot use 'none' with other component
define void @test(ptr captures(address, none) %p) {
  ret void
}

;--- none-before.ll

; CHECK-NONE-BEFORE: <stdin>:[[@LINE+1]]:38: error: cannot use 'none' with other component
define void @test(ptr captures(none, address) %p) {
  ret void
}

;--- non-pointer-type.ll

; CHECK-NON-POINTER-TYPE: Attribute 'captures(none)' applied to incompatible type!
define void @test(i32 captures(none) %p) {
  ret void
}
