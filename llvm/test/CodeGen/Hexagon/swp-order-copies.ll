; RUN: llc -march=hexagon < %s | FileCheck %s

; Test that the instruction ordering code in the pipeliner fixes up dependences
; between post-increment register definitions and uses so that the register
; allocator does not allocate an additional register. The following test case
; should generate a single packet.

; CHECK: loop0(.LBB0_[[LOOP:.]],
; CHECK: .LBB0_[[LOOP]]:
; CHECK: {
; CHECK-NOT: {
; CHECK: :endloop0

define void @test(i64* nocapture %v1, i64 %v2, i32 %len) local_unnamed_addr #0 {
entry:
  %cmp7 = icmp sgt i32 %len, 0
  br i1 %cmp7, label %for.body, label %for.end

for.body:
  %arrayidx.phi = phi i64* [ %arrayidx.inc, %for.body ], [ %v1, %entry ]
  %i.08 = phi i32 [ %inc, %for.body ], [ 0, %entry ]
  %0 = load i64, i64* %arrayidx.phi, align 8
  %1 = tail call i64 @llvm.hexagon.M2.mmpyul.rs1(i64 %0, i64 %v2)
  store i64 %1, i64* %arrayidx.phi, align 8
  %inc = add nuw nsw i32 %i.08, 1
  %exitcond = icmp eq i32 %inc, %len
  %arrayidx.inc = getelementptr i64, i64* %arrayidx.phi, i32 1
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

declare i64 @llvm.hexagon.M2.mmpyul.rs1(i64, i64) #1

attributes #0 = { nounwind "target-cpu"="hexagonv60" }
attributes #1 = { nounwind readnone }
