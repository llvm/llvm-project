;   Ensure t.a.err is non-empty.
; RUN: echo foo > %t.a.err
; RUN: llvm-link %s %S/Inputs/datalayout-a.ll -S -o - 2>>%t.a.err
; RUN: FileCheck --check-prefix=WARN-A %s < %t.a.err

; RUN: llvm-link %s %S/Inputs/datalayout-b.ll -S -o - 2>%t.b.err
; RUN: cat %t.b.err | FileCheck --check-prefix=WARN-B %s

; A module with no data layout has made no ABI commitments, so linking it
; into a module that has one should not warn, matching how an empty source
; target triple is treated.
;   Ensure t.c.err is non-empty.
; RUN: echo foo > %t.c.err
; RUN: llvm-link %S/Inputs/datalayout-b.ll %S/Inputs/datalayout-empty.ll -S -o - 2>>%t.c.err
; RUN: FileCheck --check-prefix=WARN-C %s < %t.c.err

target datalayout = "e"


; WARN-A-NOT: warning

; WARN-B: warning: Linking two modules of different data layouts:

; WARN-C-NOT: warning
