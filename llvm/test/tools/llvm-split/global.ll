; RUN: llvm-split -o %t %s
; RUN: llvm-dis -o - %t0 | FileCheck --check-prefix=CHECK0 %s
; RUN: llvm-dis -o - %t1 | FileCheck --check-prefix=CHECK1 %s

; CHECK0: @foo = global ptr @bar
; CHECK1: @foo = external global ptr
@foo = global ptr @bar

; CHECK0: @bar = external global ptr
; CHECK1: @bar = global ptr @foo
@bar = global ptr @foo
