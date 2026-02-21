; RUN: split-file %s %t

;--- common.ll
; RUN: not llvm-as %t/common.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=CHECK-COMMON
$v = comdat any
@v = common global i32 0, comdat($v)
; CHECK-COMMON: 'common' global may not be in a Comdat!

;--- private.ll
; RUN: llvm-as %t/private.ll -o /dev/null
; RUN: opt -mtriple=x86_64-unknown-linux %t/private.ll -o /dev/null
; RUN: not opt -mtriple=x86_64-pc-win32 %t/private.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=CHECK-PRIVATE
$v = comdat any
@v = private global i32 0, comdat($v)
; CHECK-PRIVATE: comdat global value has private linkage

;--- noleader.ll
; RUN: llvm-as %t/noleader.ll -o /dev/null
; RUN: opt -mtriple=x86_64-unknown-linux %t/noleader.ll -o /dev/null
; RUN: not opt -mtriple=x86_64-pc-win32 %t/noleader.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=CHECK-NOLEADER

$v = comdat any
@unrelated = internal global i32 0, comdat($v)
; CHECK-NOLEADER: COFF comdats must have a defined global value with the same name

;--- undefined.ll
; RUN: llvm-as %t/undefined.ll -o /dev/null
; RUN: opt -mtriple=x86_64-unknown-linux %t/undefined.ll -o /dev/null
; RUN: not opt -mtriple=x86_64-pc-win32 %t/undefined.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=CHECK-UNDEFINED

$v = comdat any
@v = external global i32
@unrelated = internal global i32 0, comdat($v)
; CHECK-UNDEFINED: COFF comdats must have a defined global value with the same name

;--- largest.ll
; RUN: llvm-as %t/largest.ll -o /dev/null
; This used to be invalid, but now it's valid.  Ensure the verifier
; doesn't reject it.

$v = comdat largest
