; RUN: opt -S  -dxil-op-lower  -mtriple=dxil-pc-shadermodel6.3-compute %s | FileCheck %s

define void @main() {
entry:
; CHECK: call i32 @dx.op.waveGetLaneIndex(i32 111)
  %0 = call i32 @llvm.dx.wave.getlaneindex()
  ret void
}

; CHECK: declare i32 @dx.op.waveGetLaneIndex(i32) #[[#ATTR0:]]
; CHECK: attributes #[[#ATTR0]] = { nounwind memory(read) }

declare i32 @llvm.dx.wave.getlaneindex()
