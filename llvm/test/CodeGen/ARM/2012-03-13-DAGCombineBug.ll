; RUN: llc < %s -mtriple=thumbv7-apple-ios | FileCheck %s
; rdar://11035895

; DAG combine incorrectly optimize (i32 vextract (v4i16 load $addr), c) to
; (i16 load $addr+c*sizeof(i16)). It should have issued an extload instead. i.e.
; (i32 extload $addr+c*sizeof(i16)
define void @test_hi_short3(ptr nocapture %srcA, ptr nocapture %dst) nounwind {
entry:
; CHECK: vst1.32
  %0 = load <3 x i16> , ptr %srcA, align 8
  %1 = shufflevector <3 x i16> %0, <3 x i16> undef, <2 x i32> <i32 2, i32 undef>
  store <2 x i16> %1, ptr %dst, align 4
  ret void
}

