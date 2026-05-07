; RUN: not llc < %s -mtriple=x86_64-unknown-linux-gnu 2>&1 | FileCheck %s
; Issue #196217

; CHECK: error: scaled index register cannot be negated
define i64 @minus_imm_times_reg(i64 %a, i64 %b) {
  %r = call i64 asm sideeffect inteldialect "mov $0, [$0 - 2 * $1]", "=r,r,r"(i64 %a, i64 %b)
  ret i64 %r
}

; CHECK: error: scaled index register cannot be negated
define i64 @minus_imm_times_reg_with_disp(i64 %a, i64 %b) {
  %r = call i64 asm sideeffect inteldialect "mov $0, [$0 - 2 * $1 + 8]", "=r,r,r"(i64 %a, i64 %b)
  ret i64 %r
}

; CHECK: error: scaled index register cannot be negated
define i64 @plus_then_minus_imm_times_reg(i64 %a, i64 %b) {
  %r = call i64 asm sideeffect inteldialect "mov $0, [$0 + 8 - 4 * $1]", "=r,r,r"(i64 %a, i64 %b)
  ret i64 %r
}

; CHECK: error: scaled index register cannot be negated
define i64 @minus_one_times_reg(i64 %a, i64 %b) {
  %r = call i64 asm sideeffect inteldialect "mov $0, [$0 - 1 * $1]", "=r,r,r"(i64 %a, i64 %b)
  ret i64 %r
}
