; RUN: split-file %s %t
; RUN: not llvm-as -disable-output < %t/invalid_prop.ll 2>&1 | FileCheck %t/invalid_prop.ll
; RUN: not llvm-as -disable-output < %t/missing_value1.ll 2>&1 | FileCheck %t/missing_value1.ll
; RUN: not llvm-as -disable-output < %t/missing_value2.ll 2>&1 | FileCheck %t/missing_value2.ll
; RUN: not llvm-as -disable-output < %t/missing_paren.ll 2>&1 | FileCheck %t/missing_paren.ll

;--- invalid_prop.ll
; CHECK: unknown property name
module asm(foo: "bar")
    "asm"

;--- missing_value1.ll
; CHECK: expected property name followed by ':'
module asm(target_features)
    "asm"

;--- missing_value2.ll
; CHECK: expected string constant
module asm(target_features:)
    "asm"

;--- missing_paren.ll
; CHECK: expected ',' or ')'
module asm(target_features: "bar"
    "asm"
