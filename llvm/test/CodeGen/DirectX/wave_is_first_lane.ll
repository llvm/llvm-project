; RUN: opt -S  -dxil-op-lower  -mtriple=dxil-pc-shadermodel6.3-compute %s | FileCheck %s

define void @main() #0 {
entry:
; CHECK: call i1 @dx.op.waveIsFirstLane(i32 110)
  %0 = call i1 @llvm.dx.wave.is.first.lane()
  ret void
}

; CHECK-NOT: attributes {{.*}} memory(none)

declare i1 @llvm.dx.wave.is.first.lane() #1

attributes #0 = { convergent norecurse "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #1 = { convergent nocallback nofree nosync nounwind willreturn }
