; RUN: llc < %s -mtriple=x86_64-pc-windows-msvc | FileCheck %s

; CHECK:     .section .gfids$y
; CHECK:     .symidx alias
; CHECK-NOT: .symidx calledalias
; CHECK-NOT: .symidx func


define void @func() {
  ret void
}

@alias = alias ptr, ptr @func

; This makes @alias a potential indirect call target.
; The aliasee (@func) is not considered as such.
@ptrs = global [1 x ptr] [ptr @alias]

@calledalias = alias ptr, ptr @func

define void @caller() {
  ; A direct call does not make the alias an indirect call target.
  call void @calledalias()
  ret void
}

!llvm.module.flags = !{!0}
!0 = !{i32 2, !"cfguard", i32 2}
