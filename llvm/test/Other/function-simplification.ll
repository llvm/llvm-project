; RUN: opt -passes='function-simplification<O1>' -debug-pass-manager -disable-output < %s 2>&1 | FileCheck %s --check-prefix=O1
; RUN: opt -passes='function-simplification<O2>' -debug-pass-manager -disable-output < %s 2>&1 | FileCheck %s --check-prefix=O23SZ
; RUN: opt -passes='function-simplification<O3>' -debug-pass-manager -disable-output < %s 2>&1 | FileCheck %s --check-prefix=O23SZ
; RUN: opt -passes='function-simplification<Os>' -debug-pass-manager -disable-output < %s 2>&1 | FileCheck %s --check-prefix=O23SZ
; RUN: opt -passes='function-simplification<Oz>' -debug-pass-manager -disable-output < %s 2>&1 | FileCheck %s --check-prefix=O23SZ
; RUN: not opt -passes='function-simplification<O0>' -disable-output < %s 2>&1 | FileCheck %s --check-prefix=O0

; O1: Running pass: EarlyCSEPass
; O1-NOT: Running pass: GVNPass

; O23SZ: Running pass: EarlyCSEPass
; O23SZ: Running pass: GVNPass

; O0: invalid function-simplification parameter 'O0'

define void @f() {
  ret void
}