; RUN: rm -rf %t && split-file %s %t
; RUN: not llvm-link -S %t/1.ll %t/1-aux.ll 2>&1 | FileCheck %s

; CHECK: error: Linking COMDATs named 'foo': nodeduplicate has been violated!

; RUN: not llvm-link -S %t/2.ll %t/2-aux.ll 2>&1 | FileCheck %s --check-prefix=CHECK2
; RUN: not llvm-link -S %t/2-aux.ll %t/2.ll 2>&1 | FileCheck %s --check-prefix=CHECK2

; CHECK2: error: Linking COMDATs named 'foo'

; RUN: not llvm-link -S %t/non-var.ll %t/non-var.ll 2>&1 | FileCheck %s --check-prefix=NONVAR

; NONVAR: error: Linking COMDATs named 'foo': nodeduplicate has been violated!

;--- 1.ll
$foo = comdat nodeduplicate
@foo = global i64 43, comdat($foo)

;--- 1-aux.ll
$foo = comdat nodeduplicate
@foo = global i64 43, comdat($foo)

;--- 2.ll
$foo = comdat nodeduplicate
@foo = global i64 2, section "data", comdat($foo), align 8
@bar = weak global i64 0, section "cnts", comdat($foo)
@qux = weak_odr global i64 4, comdat($foo)

;--- 2-aux.ll
$foo = comdat nodeduplicate
@foo = weak global i64 0, section "data", comdat($foo)
@bar = dso_local global i64 3, section "cnts", comdat($foo), align 16
@fred = linkonce global i64 5, comdat($foo)

;--- non-var.ll
$foo = comdat nodeduplicate
define weak void @foo() comdat {
  ret void
}
