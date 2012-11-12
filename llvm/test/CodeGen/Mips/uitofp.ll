; RUN: llc -march=mips -mattr=+single-float < %s

define void @f0() nounwind {
entry:
  %b = alloca i32, align 4
  %a = alloca float, align 4
  store volatile i32 1, i32* %b, align 4
  %0 = load volatile i32* %b, align 4
  %conv = uitofp i32 %0 to float
  store float %conv, float* %a, align 4
  ret void
}
