; RUN: not llc -mtriple=sparc <%s 2>&1 | FileCheck %s
; RUN: not llc -mtriple=sparcv9 <%s 2>&1 | FileCheck %s

; CHECK: error: could not allocate input reg for constraint '{f32}'
; CHECK: error: could not allocate input reg for constraint '{f21}'
; CHECK: error: could not allocate input reg for constraint '{f38}'
define void @test_constraint_float_reg() {
entry:
  tail call void asm sideeffect "fadds $0,$1,$2", "{f32},{f0},{f0}"(float 6.0, float 7.0, float 8.0)
  tail call void asm sideeffect "faddd $0,$1,$2", "{f21},{f0},{f0}"(double 9.0, double 10.0, double 11.0)
  tail call void asm sideeffect "faddq $0,$1,$2", "{f38},{f0},{f0}"(fp128 0xL0, fp128 0xL0, fp128 0xL0)
  ret void
}

; CHECK: <unknown>:0: error: Hi part of pair should point to an even-numbered register
; CHECK: <unknown>:0: error: (note that in some cases it might be necessary to manually bind the input/output registers instead of relying on automatic allocation)

define i64 @test_twinword_error(){
  %1 = tail call i64 asm sideeffect "rd %asr5, ${0:L} \0A\09 srlx ${0:L}, 32, ${0:H}", "={i1}"()
  ret i64 %1
}

; CHECK: error: value out of range for constraint 'J'
; CHECK: error: value out of range for constraint 'K'
; CHECK: error: value out of range for constraint 'L'
; CHECK: error: value out of range for constraint 'M'
; CHECK: error: value out of range for constraint 'N'
; CHECK: error: value out of range for constraint 'O'
define void @test_constraint_imm_out_of_range() {
entry:
  tail call void asm sideeffect "", "J"(i32 1)
  tail call void asm sideeffect "", "K"(i32 4097)
  tail call void asm sideeffect "", "L"(i32 1024)
  tail call void asm sideeffect "", "M"(i32 512)
  tail call void asm sideeffect "", "N"(i32 -1024)
  tail call void asm sideeffect "", "O"(i32 4095)
  ret void
}
