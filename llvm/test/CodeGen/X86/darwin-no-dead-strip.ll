; RUN: llc < %s | FileCheck %s

target datalayout = "e-p:32:32"
target triple = "i686-apple-darwin8.7.2"

@x = weak global i32 0
; CHECK: .no_dead_strip	_x

@"\01Ly" = private global i8 0
; CHECK: no_dead_strip Ly

@llvm.used = appending global [2 x ptr] [ ptr @x,
            ptr @"\01Ly" ]
