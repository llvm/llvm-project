; RUN: llc < %s -mtriple=arm64-none-linux-gnu -mattr=+neon -O2 | FileCheck %s

; st2 must before two ldrb.
; The situation that put one ldrb before st2 because of the conservative memVT set for st2lane,
; which lead to basic-aa goes wrong.

define dso_local i32 @test_vst2_lane_u8([2 x <8 x i8>] %vectors.coerce) local_unnamed_addr {
; CHECK-LABEL:   test_vst2_lane_u8:
; CHECK:         st2 { v[[V1:[0-9]+]].b, v[[V2:[0-9]+]].b }[6], [x8]
; CHECK-NEXT:    umov w[[W1:[0-9]+]], v[[V12:[0-9]+]].b[6]
; CHECK-NEXT:    ldrb w[[W2:[0-9]+]], [sp, #12]
; CHECK-NEXT:    ldrb w[[W2:[0-9]+]], [sp, #13]
entry:
  %temp = alloca [2 x i8], align 4
  %vectors.coerce.fca.0.extract = extractvalue [2 x <8 x i8>] %vectors.coerce, 0
  %vectors.coerce.fca.1.extract = extractvalue [2 x <8 x i8>] %vectors.coerce, 1
  call void @llvm.lifetime.start.p0(i64 2, ptr nonnull %temp) #4
  call void @llvm.aarch64.neon.st2lane.v8i8.p0(<8 x i8> %vectors.coerce.fca.0.extract, <8 x i8> %vectors.coerce.fca.1.extract, i64 6, ptr nonnull %temp)
  %0 = load i8, ptr %temp, align 4
  %vget_lane = extractelement <8 x i8> %vectors.coerce.fca.0.extract, i64 6
  %cmp8.not = icmp ne i8 %0, %vget_lane
  %arrayidx3.1 = getelementptr inbounds [2 x i8], ptr %temp, i64 0, i64 1
  %1 = load i8, ptr %arrayidx3.1, align 1
  %vget_lane.1 = extractelement <8 x i8> %vectors.coerce.fca.1.extract, i64 6
  %cmp8.not.1 = icmp ne i8 %1, %vget_lane.1
  %or.cond = select i1 %cmp8.not, i1 true, i1 %cmp8.not.1
  %cmp.lcssa = zext i1 %or.cond to i32
  call void @llvm.lifetime.end.p0(i64 2, ptr nonnull %temp) #4
  ret i32 %cmp.lcssa
}

declare void @llvm.lifetime.start.p0(i64 immarg, ptr nocapture) #2
declare void @llvm.aarch64.neon.st2lane.v8i8.p0(<8 x i8>, <8 x i8>, i64, ptr nocapture) #2
declare void @llvm.lifetime.end.p0(i64 immarg, ptr nocapture) #2
