; RUN: llc -O1 -mtriple=riscv64 -filetype=null < %s
; REQUIRES: riscv-registered-target
; Test that CodeGenPrepare doesn't crash with asm goto

define void @test_asm_goto_crash(i32 %x) {
entry:
  switch i32 %x, label %default [
    i32 0, label %indirect
  ]
indirect:
  %target = phi ptr [ label %default, label %entry ]
  callbr void asm sideeffect "j ${0:l}", "X"(ptr blockaddress(@test_asm_goto_crash, %default))
          to label %normal [label %default]
normal:
  ret void
default:
  ret void
}