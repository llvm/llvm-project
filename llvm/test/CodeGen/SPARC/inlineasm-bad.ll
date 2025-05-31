; RUN: not llc -mtriple=sparc <%s 2>&1 | FileCheck %s
; RUN: not llc -mtriple=sparcv9 <%s 2>&1 | FileCheck %s

; CHECK: error: couldn't allocate input reg for constraint '{f32}'
; CHECK: error: couldn't allocate input reg for constraint '{f21}'
; CHECK: error: couldn't allocate input reg for constraint '{f38}'
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
