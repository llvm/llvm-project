; RUN: split-file %s %t.dir
; RUN: not llvm-link %t.dir/global.ll %t.dir/global.ll -S -o - 2>&1 | FileCheck %s
; RUN: llvm-link %t.dir/global.ll %t.dir/weak.ll -S -o - 2>&1
; RUN: llvm-link %t.dir/weak.ll %t.dir/global.ll -S -o - 2>&1
; RUN: llvm-link %t.dir/weak.ll %t.dir/weak.ll -S -o - 2>&1

;--- global.ll
$foo = comdat nodeduplicate
@foo = global i64 43, comdat($foo)
; CHECK: Linking COMDATs named 'foo': nodeduplicate has been violated!

;--- weak.ll
$foo = comdat nodeduplicate
@foo = weak global i64 43, comdat($foo)
