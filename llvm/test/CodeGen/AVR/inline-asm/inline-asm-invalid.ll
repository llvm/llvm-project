; RUN: not llc < %s -mtriple=avr -no-integrated-as 2>&1 | FileCheck %s
; RUN: not llc < %s -mtriple=avr -mcpu=avr6 -filetype=obj 2>&1 \
; RUN:     FileCheck %s --check-prefix=AVR6

define void @foo(i16 %a) {
  ; CHECK: error: invalid operand in inline asm: 'jl ${0:l}'
  %i.addr = alloca i32, align 4
  call void asm sideeffect "jl ${0:l}", "*m"(i32* elementtype(i32) %i.addr)

  ret void
}

define void @foo1() {
  ; CHECK: error: invalid operand in inline asm: ';; ${0:C}'
  call i16 asm sideeffect ";; ${0:C}", "=d"()
  ret void
}

define void @foo2() {
  ; AVR6: error: expected either Y or Z register
  call void asm sideeffect "ldd r24, X+2", ""()
  ret void
}
