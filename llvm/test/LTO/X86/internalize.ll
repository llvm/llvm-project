; RUN: opt %s -o %t1.bc

; RUN: llvm-lto %t1.bc -o %t1.save.opt  --exported-symbol=_foo -save-merged-module -O0
; RUN: llvm-dis < %t1.save.opt.merged.bc | FileCheck %s --check-prefix=INTERNALIZE

; Test the enable-lto-internalization option by setting it to false.
; This makes sure internalization does not happen.
; RUN: llvm-lto %t1.bc -enable-lto-internalization=false -o %t1.save.opt  \
; RUN:                 --exported-symbol=_foo -save-merged-module -O0
; RUN: llvm-dis < %t1.save.opt.merged.bc | FileCheck %s --check-prefix=INTERNALIZE-OPTION-DISABLE

; RUN: llvm-lto2 run %t1.bc -o %t.o -save-temps \
; RUN:     -r=%t1.bc,_foo,pxl \
; RUN:     -r=%t1.bc,_bar,pl
; RUN: llvm-dis < %t.o.0.2.internalize.bc | FileCheck  %s --check-prefix=INTERNALIZE2

; Test the enable-lto-internalization option by setting it to false.
; This makes sure internalization does not happen in runRegularLTO().
; RUN: llvm-lto2 run %t1.bc -o %t.o -save-temps -enable-lto-internalization=false \
; RUN:     -r=%t1.bc,_foo,pxl \
; RUN:     -r=%t1.bc,_bar,pl
; RUN: llvm-dis < %t.o.0.2.internalize.bc | FileCheck  %s --check-prefix=INTERNALIZE2-OPTION-DISABLE

; INTERNALIZE: define void @foo
; INTERNALIZE: define internal void @bar
; INTERNALIZE-OPTION-DISABLE: define void @foo
; INTERNALIZE-OPTION-DISABLE: define void @bar
; INTERNALIZE2: define dso_local void @foo
; INTERNALIZE2: define internal void @bar
; INTERNALIZE2-OPTION-DISABLE: define dso_local void @foo
; INTERNALIZE2-OPTION-DISABLE: define dso_local void @bar

target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.11.0"

define void @foo() {
    call void @bar()
    ret void
}
define void @bar() {
    ret void
}
