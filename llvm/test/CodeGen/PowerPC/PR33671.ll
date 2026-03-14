; Function Attrs: norecurse nounwind
; RUN: llc -mtriple=powerpc64le-unknown-unknown -mcpu=pwr9 < %s | FileCheck %s
define void @test1(ptr nocapture readonly %arr, ptr nocapture %arrTo) {
entry:
  %arrayidx = getelementptr inbounds i32, ptr %arrTo, i64 4
  %arrayidx1 = getelementptr inbounds i32, ptr %arr, i64 4
  %0 = load <4 x i32>, ptr %arrayidx1, align 16
  store <4 x i32> %0, ptr %arrayidx, align 16
  ret void
; CHECK-LABEL: test1
; CHECK: lxv [[LD:[0-9]+]], 16(3)
; CHECK: stxv [[LD]], 16(4)
}

; Function Attrs: norecurse nounwind
define void @test2(ptr nocapture readonly %arr, ptr nocapture %arrTo) {
entry:
  %arrayidx = getelementptr inbounds i32, ptr %arrTo, i64 1
  %arrayidx1 = getelementptr inbounds i32, ptr %arr, i64 2
  %0 = load <4 x i32>, ptr %arrayidx1, align 16
  store <4 x i32> %0, ptr %arrayidx, align 16
  ret void
; CHECK-LABEL: test2
; CHECK: li [[REG:[0-9]+]], 8
; CHECK: lxvx [[LD:[0-9]+]], 3, [[REG]]
; CHECK: li [[REG2:[0-9]+]], 4
; CHECK: stxvx [[LD]], 4, [[REG2]]
}
