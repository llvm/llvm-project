; RUN: llc -march=hexagon -mattr=+hvxv68,+hvx-length128b < %s | FileCheck %s

; Check that ordering objects on the stack from the largest to the smallest has
; decreased the space allocated on the stack by 512 Bytes.

; CHECK: allocframe(r29,#2432):raw

define void @test(ptr nocapture readonly %arg, ptr nocapture writeonly %arg1, i32 %arg2) local_unnamed_addr {
bb:
  %shl = shl i32 %arg2, 5
  br label %bb3

bb3:
  %phi = phi i32 [ 0, %bb ], [ %add13, %bb3 ]
  %add = add i32 %phi, %shl
  %sext = sext i32 %add to i64
  %getelementptr = getelementptr float, ptr %arg, i64 %sext
  %load = load <32 x float>, ptr %getelementptr, align 4
  %fmul = fmul <32 x float> %load, <float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01>
  %fmul4 = fmul <32 x float> %load, <float 0x3FE9884620000000, float 0x3FE9884620000000, float 0x3FE9884620000000, float 0x3FE9884620000000, float 0x3FE9884620000000, float 0x3FE9884620000000, float 0x3FE9884620000000, float 0x3FE9884620000000, float 0x3FE9884620000000, float 0x3FE9884620000000, float 0x3FE9884620000000, float 0x3FE9884620000000, float 0x3FE9884620000000, float 0x3FE9884620000000, float 0x3FE9884620000000, float 0x3FE9884620000000, float 0x3FE9884620000000, float 0x3FE9884620000000, float 0x3FE9884620000000, float 0x3FE9884620000000, float 0x3FE9884620000000, float 0x3FE9884620000000, float 0x3FE9884620000000, float 0x3FE9884620000000, float 0x3FE9884620000000, float 0x3FE9884620000000, float 0x3FE9884620000000, float 0x3FE9884620000000, float 0x3FE9884620000000, float 0x3FE9884620000000, float 0x3FE9884620000000, float 0x3FE9884620000000>
  %fmul5 = fmul <32 x float> %load, <float 0x3FA2444180000000, float 0x3FA2444180000000, float 0x3FA2444180000000, float 0x3FA2444180000000, float 0x3FA2444180000000, float 0x3FA2444180000000, float 0x3FA2444180000000, float 0x3FA2444180000000, float 0x3FA2444180000000, float 0x3FA2444180000000, float 0x3FA2444180000000, float 0x3FA2444180000000, float 0x3FA2444180000000, float 0x3FA2444180000000, float 0x3FA2444180000000, float 0x3FA2444180000000, float 0x3FA2444180000000, float 0x3FA2444180000000, float 0x3FA2444180000000, float 0x3FA2444180000000, float 0x3FA2444180000000, float 0x3FA2444180000000, float 0x3FA2444180000000, float 0x3FA2444180000000, float 0x3FA2444180000000, float 0x3FA2444180000000, float 0x3FA2444180000000, float 0x3FA2444180000000, float 0x3FA2444180000000, float 0x3FA2444180000000, float 0x3FA2444180000000, float 0x3FA2444180000000>
  %fmul6 = fmul <32 x float> %load, %fmul5
  %fmul7 = fmul <32 x float> %load, %fmul6
  %fadd = fadd <32 x float> %fmul4, %fmul7
  %fmul8 = fmul <32 x float> %fadd, <float -2.000000e+00, float -2.000000e+00, float -2.000000e+00, float -2.000000e+00, float -2.000000e+00, float -2.000000e+00, float -2.000000e+00, float -2.000000e+00, float -2.000000e+00, float -2.000000e+00, float -2.000000e+00, float -2.000000e+00, float -2.000000e+00, float -2.000000e+00, float -2.000000e+00, float -2.000000e+00, float -2.000000e+00, float -2.000000e+00, float -2.000000e+00, float -2.000000e+00, float -2.000000e+00, float -2.000000e+00, float -2.000000e+00, float -2.000000e+00, float -2.000000e+00, float -2.000000e+00, float -2.000000e+00, float -2.000000e+00, float -2.000000e+00, float -2.000000e+00, float -2.000000e+00, float -2.000000e+00>
  %call = tail call <32 x float> @llvm.exp.v32f32(<32 x float> %fmul8)
  %fsub = fsub <32 x float> <float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00>, %call
  %fadd9 = fadd <32 x float> %call, <float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00>
  %fdiv = fdiv <32 x float> %fsub, %fadd9
  %fadd10 = fadd <32 x float> %fdiv, <float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00>
  %fmul11 = fmul <32 x float> %fmul, %fadd10
  %getelementptr12 = getelementptr float, ptr %arg1, i64 %sext
  store <32 x float> %fmul11, ptr %getelementptr12, align 128
  %add13 = add nuw nsw i32 %phi, 128
  %icmp = icmp ult i32 %phi, 8064
  br i1 %icmp, label %bb3, label %bb14

bb14:
  ret void
}

declare <32 x float> @llvm.exp.v32f32(<32 x float>)
