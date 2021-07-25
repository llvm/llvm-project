; RUN: not llvm-link %s %p/Inputs/comdat3.ll -S -o - 2>&1 | FileCheck %s

$foo = comdat nodeduplicate
@foo = global i64 43, comdat($foo)
; CHECK: Linking COMDATs named 'foo': nodeduplicate has been violated!
