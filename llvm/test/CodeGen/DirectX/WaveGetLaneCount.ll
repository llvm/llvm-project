; RUN: opt -S  -dxil-op-lower  -mtriple=dxil-pc-shadermodel6.3-compute %s | FileCheck %s

define void @main() {
entry:
; CHECK: call i32 @dx.op.waveGetLaneCount(i32 112) #[[#ATTR:]]
  %0 = call i32 @llvm.dx.wave.get.lane.count()
  ret void
}

; CHECK: attributes #[[#ATTR]] = {{{.*}} memory(read) {{.*}}}

declare i32 @llvm.dx.wave.get.lane.count()

