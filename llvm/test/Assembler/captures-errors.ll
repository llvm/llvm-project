; RUN: split-file --leading-lines %s %t
; RUN: not llvm-as < %t/missing-lparen.ll 2>&1 | FileCheck %s --check-prefix=CHECK-MISSING-LPAREN
; RUN: not llvm-as < %t/missing-rparen.ll 2>&1 | FileCheck %s --check-prefix=CHECK-MISSING-RPAREN
; RUN: not llvm-as < %t/missing-rparen-none.ll 2>&1 | FileCheck %s --check-prefix=CHECK-MISSING-RPAREN-NONE
; RUN: not llvm-as < %t/missing-colon.ll 2>&1 | FileCheck %s --check-prefix=CHECK-MISSING-COLON
; RUN: not llvm-as < %t/invalid-component.ll 2>&1 | FileCheck %s --check-prefix=CHECK-INVALID-COMPONENT

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

; CHECK-MISSING-RPAREN-NONE: <stdin>:[[@LINE+1]]:37: error: expected ')'
define void @test(ptr captures(none %p) {
  ret void
}

;--- missing-colon.ll

; CHECK-MISSING-COLON: <stdin>:[[@LINE+1]]:36: error: expected ':'
define void @test(ptr captures(ret address) %p) {
  ret void
}

;--- invalid-component.ll

; CHECK-INVALID-COMPONENT: <stdin>:[[@LINE+1]]:32: error: expected one of 'address', 'provenance' or 'read_provenance'
define void @test(ptr captures(foo) %p) {
  ret void
}
