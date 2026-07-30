; RUN: llc -mtriple=hexagon -fp-contract=fast < %s | FileCheck %s

; Check that "Rx-=sfmpy(Rs,Rt)" is being generated for "fsub(fmul(..))"

; CHECK: r{{[0-9]+}} -= sfmpy

%struct.matrix_params = type { ptr }

; Function Attrs: norecurse nounwind
define void @loop2_1(ptr nocapture readonly %params, i32 %col1) #0 {
entry:
  %0 = load ptr, ptr %params, align 4
  %1 = load ptr, ptr %0, align 4
  %arrayidx1 = getelementptr inbounds float, ptr %1, i32 %col1
  %2 = load float, ptr %arrayidx1, align 4
  %arrayidx3 = getelementptr inbounds ptr, ptr %0, i32 %col1
  %3 = load ptr, ptr %arrayidx3, align 4
  %4 = load float, ptr %3, align 4
  %mul = fmul float %2, %4
  %sub = fsub float %2, %mul
  %arrayidx10 = getelementptr inbounds float, ptr %3, i32 %col1
  store float %sub, ptr %arrayidx10, align 4
  ret void
}
