; RUN: opt -module-summary %s -o %t1.bc
; RUN: opt -module-summary %p/Inputs/internalize.ll -o %t2.bc
; Link in %t2.bc first to force its copy of @weak_func_nonprevailing as
; prevailing the %t1.bc copy as non-prevailing.
; RUN: llvm-lto -thinlto-action=thinlink -o %t.index.bc %t2.bc %t1.bc
; RUN: llvm-lto -thinlto-action=internalize -thinlto-index %t.index.bc %t1.bc -o - | llvm-dis -o - | FileCheck %s --check-prefix=REGULAR
; RUN: llvm-lto -thinlto-action=internalize -thinlto-index %t.index.bc %t1.bc -o -  --exported-symbol=_foo | llvm-dis -o - | FileCheck %s --check-prefix=INTERNALIZE

; Test the enable-lto-internalization option by setting it to false.
; This makes sure indices are not marked as internallinkage and therefore
; internalization does not happen.
; RUN: llvm-lto -thinlto-action=internalize -thinlto-index %t.index.bc %t1.bc \
; RUN:          -enable-lto-internalization=false --exported-symbol=_foo
; RUN: llvm-dis < %t1.bc.thinlto.internalized.bc | FileCheck %s --check-prefix=INTERNALIZE-OPTION-DISABLE

; RUN: llvm-lto2 run %t1.bc -o %t.o -save-temps \
; RUN:     -r=%t1.bc,_foo,pxl \
; RUN:     -r=%t1.bc,_bar,pl \
; RUN:     -r=%t1.bc,_linkonce_func,pl \
; RUN:     -r=%t1.bc,_weak_func_prevailing,pl \
; RUN:     -r=%t1.bc,_alias1,plx \
; RUN:     -r=%t1.bc,_weak_func_nonprevailing,l
; RUN: llvm-dis < %t.o.1.2.internalize.bc | FileCheck  %s --check-prefix=INTERNALIZE2

; Test the enable-lto-internalization option by setting it to false.
; This makes sure indices are not marked as internallinkage and therefore
; internalization does not happen.
; RUN: llvm-lto2 run %t1.bc -o %t.o -save-temps -enable-lto-internalization=false \
; RUN:     -r=%t1.bc,_foo,pxl \
; RUN:     -r=%t1.bc,_bar,pl \
; RUN:     -r=%t1.bc,_linkonce_func,pl \
; RUN:     -r=%t1.bc,_weak_func_prevailing,pl \
; RUN:     -r=%t1.bc,_alias1,plx \
; RUN:     -r=%t1.bc,_weak_func_nonprevailing,l
; RUN: llvm-dis < %t.o.1.2.internalize.bc | FileCheck  %s --check-prefix=INTERNALIZE2-OPTION-DISABLE

; REGULAR: define void @foo
; REGULAR: define void @bar
; REGULAR: define linkonce void @linkonce_func()
; INTERNALIZE: define void @foo
; INTERNALIZE: define internal void @bar
; INTERNALIZE: define internal void @linkonce_func()
; INTERNALIZE: define internal void @weak_func_prevailing()
; INTERNALIZE: define weak void @weak_func_nonprevailing()
; INTERNALIZE-OPTION-DISABLE: define void @foo
; INTERNALIZE-OPTION-DISABLE: define void @bar
; INTERNALIZE-OPTION-DISABLE: define weak void @linkonce_func()
; INTERNALIZE-OPTION-DISABLE: define weak void @weak_func_prevailing()
; INTERNALIZE-OPTION-DISABLE: define weak void @weak_func_nonprevailing()
; INTERNALIZE2: define dso_local void @foo
; INTERNALIZE2: define internal void @bar
; INTERNALIZE2: define internal void @linkonce_func()
; INTERNALIZE2: define internal void @weak_func_prevailing()
; INTERNALIZE2: define weak dso_local void @weak_func_nonprevailing()
; INTERNALIZE2-OPTION-DISABLE: define dso_local void @foo
; INTERNALIZE2-OPTION-DISABLE: define dso_local void @bar
; INTERNALIZE2-OPTION-DISABLE: define weak dso_local void @linkonce_func()
; INTERNALIZE2-OPTION-DISABLE: define weak dso_local void @weak_func_prevailing()
; INTERNALIZE2-OPTION-DISABLE: define weak dso_local void @weak_func_nonprevailing()

target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.11.0"

define void @foo() {
    call void @bar()
    ret void
}
define void @bar() {
    call void @linkonce_func()
    call void @weak_func_prevailing()
    call void @weak_func_nonprevailing()
	ret void
}
define linkonce void @linkonce_func() {
    ret void
}
define weak void @weak_func_prevailing() {
    ret void
}
; Make @weak_func_nonprevailing an aliasee to ensure it is still marked
; live and kept as a definition even when non-prevailing. We want to ensure
; this definition is not internalized.
@alias1 = hidden alias void (), void ()* @weak_func_nonprevailing
define weak void @weak_func_nonprevailing() {
    ret void
}
