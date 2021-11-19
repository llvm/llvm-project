; RUN: not llc < %s -march=avr -no-integrated-as 2>&1 | FileCheck %s

define void @foo(i16 %a) {
  ; CHECK: error: invalid operand in inline asm: 'jl ${0:l}'
  %i.addr = alloca i32, align 4
  call void asm sideeffect "jl ${0:l}", "*m"(i32* %i.addr)

  ret void
}

