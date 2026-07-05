; RUN: rm -rf %t && split-file %s %t
; RUN: not llvm-link -S %t/1.ll %t/1-aux.ll 2>&1 | FileCheck %s

; CHECK: Linking globals named 'foo': symbol multiply defined!

; RUN: llvm-link -S %t/2.ll %t/2-aux.ll | FileCheck %s --check-prefix=CHECK2
; RUN: llvm-link -S %t/2-aux.ll %t/2.ll | FileCheck %s --check-prefix=CHECK2

; CHECK2-DAG: @[[#]] = private global i64 0, section "data", comdat($foo)
; CHECK2-DAG: @[[#]] = private global i64 0, section "cnts", comdat($foo)
; CHECK2-DAG: @foo = hidden global i64 2, section "data", comdat, align 8
; CHECK2-DAG: @bar = dso_local global i64 3, section "cnts", comdat($foo), align 16
; CHECK2-DAG: @qux = weak_odr global i64 4, comdat($foo)
; CHECK2-DAG: @fred = linkonce global i64 5, comdat($foo)

; RUN: llvm-link -S %t/non-var.ll %t/non-var.ll 2>&1 | FileCheck %s --check-prefix=NONVAR

; NONVAR: linking 'foo': non-variables in comdat nodeduplicate are not handled

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
@foo = weak hidden global i64 0, section "data", comdat($foo)
@bar = dso_local global i64 3, section "cnts", comdat($foo), align 16
@fred = linkonce global i64 5, comdat($foo)

;--- non-var.ll
$foo = comdat nodeduplicate
define weak void @foo() comdat {
  ret void
}
