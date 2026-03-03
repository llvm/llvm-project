; RUN: llc %s -mtriple=x86_64-linux-gnu -o - -verify-machineinstrs \
; RUN:   | FileCheck --check-prefixes=X64 %s
; RUN: llc %s -mtriple=i686-linux-gnu -o - -verify-machineinstrs \
; RUN:   | FileCheck --check-prefix=X86 %s

define void @test1() #0 {
entry:
  ret void

; X64-LABEL: @test1
; X64: callq __fentry__
; X64: retq
; X86-LABEL: @test1
; X86: calll __fentry__
; X86: retl
}

define void @test2() #1 {
entry:
  ret void

; X64-LABEL: @test2
; X64-NOT: callq __fentry__
; X64: nopl 8(%rax,%rax)
; X64: retq
; X86-LABEL: @test2
; X86-NOT: calll __fentry__
; X86: xchgw %ax, %ax
; X86: xchgw %ax, %ax
; X86: nop
; X86: retl
}

attributes #0 = { "fentry-call"="true" }
attributes #1 = { "fentry-call"="true" "mnop-mcount" }

