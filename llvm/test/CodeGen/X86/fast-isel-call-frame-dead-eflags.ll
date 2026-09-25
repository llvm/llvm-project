; RUN: llc -mtriple=x86_64-- -O0 -stop-after=finalize-isel -o - %s | FileCheck %s

; The call frame pseudos clobber EFLAGS, but nothing reads it.

declare void @foo(i32)

define void @caller(i32 %x) {
  ; CHECK: ADJCALLSTACKDOWN64 0, 0, 0, implicit-def $rsp, implicit-def dead $eflags, implicit-def $ssp, implicit $rsp, implicit $ssp
  ; CHECK: ADJCALLSTACKUP64 0, 0, implicit-def $rsp, implicit-def dead $eflags, implicit-def $ssp, implicit $rsp, implicit $ssp
  call void @foo(i32 %x)
  ret void
}
