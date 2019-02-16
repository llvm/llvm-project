; Function Attrs: norecurse nounwind
; RUN: llc -mtriple=powerpc64le-unknown-unknown -mcpu=pwr9 < %s | FileCheck %s
define void @test1(i32* nocapture readonly %arr, i32* nocapture %arrTo) {
entry:
  %arrayidx = getelementptr inbounds i32, i32* %arrTo, i64 4
  %0 = bitcast i32* %arrayidx to <4 x i32>*
  %arrayidx1 = getelementptr inbounds i32, i32* %arr, i64 4
  %1 = bitcast i32* %arrayidx1 to <4 x i32>*
  %2 = load <4 x i32>, <4 x i32>* %1, align 16
  store <4 x i32> %2, <4 x i32>* %0, align 16
  ret void
; CHECK-LABEL: test1
; CHECK: lxv [[LD:[0-9]+]], 16(3)
; CHECK: stxv [[LD]], 16(4)
}

; Function Attrs: norecurse nounwind
define void @test2(i32* nocapture readonly %arr, i32* nocapture %arrTo) {
entry:
  %arrayidx = getelementptr inbounds i32, i32* %arrTo, i64 1
  %0 = bitcast i32* %arrayidx to <4 x i32>*
  %arrayidx1 = getelementptr inbounds i32, i32* %arr, i64 2
  %1 = bitcast i32* %arrayidx1 to <4 x i32>*
  %2 = load <4 x i32>, <4 x i32>* %1, align 16
  store <4 x i32> %2, <4 x i32>* %0, align 16
  ret void
; CHECK-LABEL: test2
; CHECK: addi 3, 3, 8
; CHECK: lxvx [[LD:[0-9]+]], 0, 3
; CHECK: addi [[REG:[0-9]+]], 4, 4
; CHECK: stxvx [[LD]], 0, [[REG]] 
}
