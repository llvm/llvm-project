; RUN: llvm-split --opaque-pointers=0 -o %t %s
; RUN: llvm-dis --opaque-pointers=0 -o - %t0 | FileCheck --check-prefix=CHECK0 %s
; RUN: llvm-dis --opaque-pointers=0 -o - %t1 | FileCheck --check-prefix=CHECK1 %s

; CHECK0: @foo = global i8* bitcast
; CHECK1: @foo = external global i8*
@foo = global i8* bitcast (i8** @bar to i8*)

; CHECK0: @bar = external global i8*
; CHECK1: @bar = global i8* bitcast
@bar = global i8* bitcast (i8** @foo to i8*)
