; simple_add.ll

; add two "register" values (function arguments in SSA form)
define i32 @add_regs(i32 %a, i32 %b) {
entry:
  %sum = add i32 %a, %b
  ret i32 %sum
}

; add an immediate constant to a register value
define i32 @add_imm_reg(i32 %a) {
entry:
  %sum = add i32 %a, 5
  ret i32 %sum
}

; optional main so the file can be linked into an executable too
define i32 @main() {
entry:
  %r1 = call i32 @add_regs(i32 10, i32 20)   ; 30
  %r2 = call i32 @add_imm_reg(i32 %r1)       ; 35
  ret i32 %r2
}