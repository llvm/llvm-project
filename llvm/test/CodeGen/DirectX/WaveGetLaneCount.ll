; RUN: opt -S  -dxil-op-lower  -mtriple=dxil-pc-shadermodel6.3-compute %s | FileCheck %s

define void @main() {
entry:
; CHECK: call i32 @dx.op.waveGetLaneCount(i32 112)
  %0 = call i32 @llvm.dx.wave.get.lane.count()
  ret void
}

; CHECK: declare i32 @dx.op.waveGetLaneCount(i32) #[[#ATTR:]]
; CHECK: attributes #[[#ATTR]] = { nounwind memory(read) }

declare i32 @llvm.dx.wave.get.lane.count()
