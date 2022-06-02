; RUN: opt -module-summary %s -o %t1.bc
; RUN: opt -module-summary %p/Inputs/reference_non_importable.ll -o %t2.bc

; RUN: llvm-lto2 run %t1.bc %t2.bc -o %t.o -save-temps \
; RUN:     -r=%t1.bc,_foo,pxl \
; RUN:     -r=%t1.bc,_b,pxl \
; RUN:     -r=%t2.bc,_main,pxl \
; RUN:     -r=%t2.bc,_foo,xl




target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.11.0"

; We shouldn't promote the private because it has a section
; RUN: llvm-dis < %t.o.1.2.internalize.bc | FileCheck  %s --check-prefix=PROMOTE
; PROMOTE: @a = private global i8 0, section "__TEXT,__cstring,cstring_literals"
@a = private global i8 0, section "__TEXT,__cstring,cstring_literals"
@b = global i8 *@a


; We want foo to be imported in the main module!
; RUN: llvm-dis < %t.o.2.3.import.bc  | FileCheck  %s --check-prefix=IMPORT
; IMPORT: define available_externally dso_local ptr @foo()
define i8 **@foo() {
	ret i8 **@b
}
