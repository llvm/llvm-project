; RUN: rm -rf %t && split-file %s %t
; RUN: not llvm-link %t/1.ll %t/1-aux.ll -S -o - 2>&1 | FileCheck %s

;--- 1.ll
$foo = comdat nodeduplicate
@foo = global i64 43, comdat($foo)
; CHECK: Linking COMDATs named 'foo': nodeduplicate has been violated!

;--- 1-aux.ll
$foo = comdat nodeduplicate
@foo = global i64 43, comdat($foo)
