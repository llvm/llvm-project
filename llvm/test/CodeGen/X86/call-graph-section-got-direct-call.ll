;; Tests that direct calls that go through the GOT (e.g. `nonlazybind` callees or
;; `-fno-plt`) are recorded as direct callees in the .llvm.callgraph section
;; instead of being dropped as indirect calls without type identifiers.

; RUN: llc -mtriple=x86_64-unknown-linux -relocation-model=pic --call-graph-section -o - < %s | FileCheck %s

declare !callgraph !0 void @callee() #0

declare !callgraph !0 void @tail_callee() #0

define void @caller() !callgraph !0 {
entry:
  call void @callee()
  ret void
}

define void @tail_caller() !callgraph !0 {
entry:
  tail call void @tail_callee()
  ret void
}

attributes #0 = { nonlazybind }

!0 = !{!"_ZTSFvE"}

; CHECK-LABEL: caller:
; CHECK: callq *callee@GOTPCREL(%rip)
; CHECK: .section .llvm.callgraph,"o",@llvm_call_graph,.text
;; Version
; CHECK-NEXT: .byte 0
;; Flags -- potential indirect target with direct callees
; CHECK-NEXT: .byte 3
;; Function Entry PC
; CHECK-NEXT: .quad caller
;; Function type ID
; CHECK-NEXT: .quad 6588678392271548388
;; Number of unique direct callees
; CHECK-NEXT: .byte 1
;; Direct callee reached through the GOT
; CHECK-NEXT: .quad callee

; CHECK-LABEL: tail_caller:
; CHECK: jmpq *tail_callee@GOTPCREL(%rip)
; CHECK: .section .llvm.callgraph,"o",@llvm_call_graph,.text
; CHECK-NEXT: .byte 0
; CHECK-NEXT: .byte 3
; CHECK-NEXT: .quad tail_caller
; CHECK-NEXT: .quad 6588678392271548388
; CHECK-NEXT: .byte 1
;; Direct tail callee reached through the GOT
; CHECK-NEXT: .quad tail_callee
