; Weak references (extern_weak) must be weak in the object file: the indirect
; symbol through which the address of a function is taken, and the part
; reference for external data. Otherwise the binder fails when the symbol
; does not exist.
; RUN: llc < %s -mtriple=s390x-ibm-zos | FileCheck %s
; RUN: llc < %s -mtriple=s390x-ibm-zos --filetype=obj | \
; RUN:   od -Ax -tx1 -v | FileCheck --check-prefix=CHECKOBJ --ignore-case %s

; CHECK-DAG: WXTRN wf@indirect
; CHECK-DAG: WXTRN wf{{ *$}}
; CHECK-NOT: {{^ EXTRN wf}}

; The last bytes of the ESD records: behavioral attributes 4-9 (binding
; strength 1 = weak in byte 4), name length and name.
; PR wv (external data):
; CHECKOBJ: {{^[0-9a-f]+}} 01 04 24 00 00 00 00 02 a6 a5
; ER wf@indirect (external name wf, indirect reference):
; CHECKOBJ: {{^[0-9a-f]+}} 01 14 20 00 00 00 00 02 a6 86
; ER wf:
; CHECKOBJ: {{^[0-9a-f]+}} 01 04 20 00 00 00 00 02 a6 86

@wv = extern_weak global i32

declare extern_weak void @wf()

define ptr @getv() {
  ret ptr @wv
}

define ptr @getf() {
  ret ptr @wf
}

define void @callf() {
  call void @wf()
  ret void
}
