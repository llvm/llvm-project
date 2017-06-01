; RUN: llc -march=hexagon -fp-contract=fast < %s | FileCheck %s

; Check that "Rx-=sfmpy(Rs,Rt)" is being generated for "fsub(fmul(..))"

; CHECK: r{{[0-9]+}} -= sfmpy

%struct.matrix_params = type { float** }

; Function Attrs: norecurse nounwind
define void @loop2_1(%struct.matrix_params* nocapture readonly %params, i32 %col1) #0 {
entry:
  %matrixA = getelementptr inbounds %struct.matrix_params, %struct.matrix_params* %params, i32 0, i32 0
  %0 = load float**, float*** %matrixA, align 4
  %1 = load float*, float** %0, align 4
  %arrayidx1 = getelementptr inbounds float, float* %1, i32 %col1
  %2 = load float, float* %arrayidx1, align 4
  %arrayidx3 = getelementptr inbounds float*, float** %0, i32 %col1
  %3 = load float*, float** %arrayidx3, align 4
  %4 = load float, float* %3, align 4
  %mul = fmul float %2, %4
  %sub = fsub float %2, %mul
  %arrayidx10 = getelementptr inbounds float, float* %3, i32 %col1
  store float %sub, float* %arrayidx10, align 4
  ret void
}
