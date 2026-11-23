; RUN: split-file %s %t

;--- common.ll
; RUN: not llvm-as %t/common.ll -disable-output 2>&1 | FileCheck %s --check-prefix=CHECK-COMMON
$v = comdat any
@v = common global i32 0, comdat($v)
; CHECK-COMMON: 'common' global may not be in a Comdat!

;--- private.ll
; RUN: llvm-as %t/private.ll -disable-output
; RUN: opt -mtriple=x86_64-unknown-linux %t/private.ll -disable-output
; RUN: not opt -mtriple=x86_64-pc-win32 %t/private.ll -disable-output 2>&1 | FileCheck %s --check-prefix=CHECK-PRIVATE
$v = comdat any
@v = private global i32 0, comdat($v)
; CHECK-PRIVATE: comdat global value has private linkage

;--- noleader.ll
; RUN: llvm-as %t/noleader.ll -disable-output
; RUN: opt -mtriple=x86_64-unknown-linux %t/noleader.ll -disable-output
; RUN: not opt -mtriple=x86_64-pc-win32 %t/noleader.ll -disable-output 2>&1 | FileCheck %s --check-prefix=CHECK-NOLEADER

$v = comdat any
@unrelated = internal global i32 0, comdat($v)
; CHECK-NOLEADER: COFF comdats must have a defined global value with the same name

;--- undefined.ll
; RUN: llvm-as %t/undefined.ll -disable-output
; RUN: opt -mtriple=x86_64-unknown-linux %t/undefined.ll -disable-output
; RUN: not opt -mtriple=x86_64-pc-win32 %t/undefined.ll -disable-output 2>&1 | FileCheck %s --check-prefix=CHECK-UNDEFINED

$v = comdat any
@v = external global i32
@unrelated = internal global i32 0, comdat($v)
; CHECK-UNDEFINED: COFF comdats must have a defined global value with the same name

;--- largest.ll
; RUN: llvm-as %t/largest.ll -disable-output
; This used to be invalid, but now it's valid.  Ensure the verifier
; doesn't reject it.

$v = comdat largest
