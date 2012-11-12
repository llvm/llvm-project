; RUN: llc -march=hexagon -mcpu=hexagonv4 < %s | FileCheck %s
; CHECK: __hexagon_addsf3
; CHECK: __hexagon_subsf3

define void @foo(float* %acc, float %num, float %num2) nounwind {
entry:
  %acc.addr = alloca float*, align 4
  %num.addr = alloca float, align 4
  %num2.addr = alloca float, align 4
  store float* %acc, float** %acc.addr, align 4
  store float %num, float* %num.addr, align 4
  store float %num2, float* %num2.addr, align 4
  %0 = load float** %acc.addr, align 4
  %1 = load float* %0
  %2 = load float* %num.addr, align 4
  %add = fadd float %1, %2
  %3 = load float* %num2.addr, align 4
  %sub = fsub float %add, %3
  %4 = load float** %acc.addr, align 4
  store float %sub, float* %4
  ret void
}
