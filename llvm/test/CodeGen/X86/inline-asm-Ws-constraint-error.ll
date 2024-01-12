; RUN: not llc -mtriple=x86_64 < %s 2>&1 | FileCheck %s

; CHECK: error: invalid operand for inline asm constraint 'Ws'
define void @test() {
entry:
  %x = alloca i32, align 4
  call void asm sideeffect "// ${0:p}", "^Ws,~{dirflag},~{fpsr},~{flags}"(ptr %x)
  ret void
}
