; RUN: llc < %s -mtriple=arm-unknown-unknown | FileCheck %s -check-prefix=NOOPTION
; RUN: llc < %s -mtriple=arm-unknown-unknown -trap-func=trap_llc | FileCheck %s -check-prefix=TRAP

; NOOPTION-LABEL: {{\_?}}foo0:
; NOOPTION: trap{{$}}

; TRAP-LABEL: {{\_?}}foo0:
; TRAP: bl {{\_?}}trap_llc

define void @foo0() {
  call void @llvm.trap()
  unreachable
}

; NOOPTION-LABEL: {{\_?}}foo1:
; NOOPTION: bl {{\_?}}trap_func_attr0

; TRAP-LABEL: {{\_?}}foo1:
; TRAP: bl {{\_?}}trap_llc

define void @foo1() {
  call void @llvm.trap() #0
  unreachable
}

; NOOPTION-LABEL: {{\_?}}foo2:
; NOOPTION: bl {{\_?}}trap_func_attr1

; TRAP-LABEL: {{\_?}}foo2:
; TRAP: bl {{\_?}}trap_llc

define void @foo2() {
  call void @llvm.trap() #1
  unreachable
}

declare void @llvm.trap() nounwind

attributes #0 = { "trap-func-name"="trap_func_attr0" }
attributes #1 = { "trap-func-name"="trap_func_attr1" }
