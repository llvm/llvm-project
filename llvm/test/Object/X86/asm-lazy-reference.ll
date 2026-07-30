; RUN: llvm-as < %s -o - | llvm-nm -  | FileCheck %s

target datalayout = "e-m:o-p:32:32-f64:32:64-f80:128-n8:16:32-S128"
target triple = "i386-apple-macosx10.8.0"

; Verify that llvm-nm handles correctly module level ASM, including "lazy_reference"

; CHECK: U .objc_class_name_Bar
; CHECK: U .objc_class_name_Foo
; CHECK: T .objc_class_name_FooSubClass

module asm "\09.objc_class_name_FooSubClass=0"
module asm "\09.globl .objc_class_name_FooSubClass"
module asm "\09.lazy_reference .objc_class_name_Foo"
module asm "\09.lazy_reference .objc_class_name_Bar"
