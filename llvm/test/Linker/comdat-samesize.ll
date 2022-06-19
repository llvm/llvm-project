; RUN: rm -rf %t && split-file %s %t
; RUN: not llvm-link %t/diff-size.ll %t/diff-size-aux.ll -S -o - 2>&1 | FileCheck %s --check-prefix=DIFF_SIZE

;--- diff-size.ll
; DIFF_SIZE: Linking COMDATs named 'foo': SameSize violated!
target datalayout = "e-m:w-p:32:32-i64:64-f80:32-n8:16:32-S32"
target triple = "i686-pc-windows-msvc"

$foo = comdat samesize
@foo = global i32 42, comdat($foo)

;--- diff-size-aux.ll
target datalayout = "e-m:w-p:32:32-i64:64-f80:32-n8:16:32-S32"
target triple = "i686-pc-windows-msvc"

$foo = comdat samesize
@foo = global i64 42, comdat($foo)
