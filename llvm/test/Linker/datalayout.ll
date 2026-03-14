;   Ensure t.a.err is non-empty.
; RUN: echo foo > %t.a.err
; RUN: llvm-link %s %S/Inputs/datalayout-a.ll -S -o - 2>>%t.a.err
; RUN: FileCheck --check-prefix=WARN-A %s < %t.a.err

; RUN: llvm-link %s %S/Inputs/datalayout-b.ll -S -o - 2>%t.b.err
; RUN: cat %t.b.err | FileCheck --check-prefix=WARN-B %s

target datalayout = "e"


; WARN-A-NOT: warning

; WARN-B: warning: Linking two modules of different data layouts:
