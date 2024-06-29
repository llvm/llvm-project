; RUN: not llc < %s -mtriple=x86_64-unknown-unknown 2>&1 | FileCheck %s

; CHECK: error: invalid operand in inline asm: 'mov %ah, ${0:h}'
define void @test1() {
entry:
  %0 = tail call i8 asm sideeffect "mov %ah, ${0:h}", "=r,~{eax},~{ebx},~{ecx},~{edx},~{dirflag},~{fpsr},~{flags}"()
  ret void
}

;CHECK: error: invalid operand in inline asm: 'vmovd ${1:k}, $0'
define i32 @foo() {
entry:
  %0 = tail call i32 asm sideeffect "vmovd ${1:k}, $0", "=r,x,~{dirflag},~{fpsr},~{flags}"(<2 x i64> <i64 240518168632, i64 240518168632>)
  ret i32 %0
}
