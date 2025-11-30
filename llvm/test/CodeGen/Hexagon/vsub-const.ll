; RUN: llc -march=hexagon -mattr=+hvxv73,+hvx-length128b < %s | FileCheck %s

; Make sure that the appropriate vadd instructions are generated when
; addtiplied with a vector of constant values.

; CHECK-LABEL: test_vadd_const1
; CHECK: [[REG0:(r[0-9]+)]] = #
; CHECK: [[VREG0:(v[0-9]+)]].b = vsplat([[REG0]])
; CHECK: v{{[0-9:]+}}.h = vadd(v{{[0-9]+}}.ub,[[VREG0]].ub)

; Function Attrs: norecurse nounwind
define dso_local void @test_vadd_const1(ptr nocapture readonly %a, ptr nocapture %r) local_unnamed_addr #0 {
entry:
  %wide.load = load <128 x i8>, ptr %a, align 1
  %0 = zext <128 x i8> %wide.load to <128 x i32>
  %1 = add nuw nsw <128 x i32> %0, <i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23>
  store <128 x i32> %1, ptr %r, align 4
  ret void
}

; CHECK-LABEL: test_vadd_const2
; CHECK: [[REG0:(r[0-9]+)]] = #-
; CHECK: [[VREG0:([0-9]+)]].h = vsplat([[REG0]])
; CHECK: [[VREG1:([0-9:])]] = v[[VREG0]]
; CHECK: v{{[0-9:]+}}.h = vadd(v{{[0-9:]+}}.h,{{v[VREG0]|v[VREG1]}}

; Function Attrs: norecurse nounwind
define dso_local void @test_vadd_const2(ptr nocapture readonly %a, ptr nocapture %r) local_unnamed_addr #0 {
entry:
  %wide.load = load <128 x i8>, ptr %a, align 1
  %0 = zext <128 x i8> %wide.load to <128 x i32>
  %1 = add nsw <128 x i32> %0, <i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23>
  store <128 x i32> %1, ptr %r, align 4
  ret void
}

; CHECK-LABEL: test_vadd_const2_1
; CHECK: [[REG0:(r[0-9]+)]] = #-270
; CHECK: [[VREG0:([0-9]+)]] = vsplat([[REG0]])
; CHECK: [[VREG1:([0-9:]+)]] = v[[VREG0]]
; CHECK: v{{[0-9:]+}}.w = vadd({{.*}}.w,{{v[VREG0]|v[VREG1]}}

; Function Attrs: norecurse nounwind
define dso_local void @test_vadd_const2_1(ptr nocapture readonly %a, ptr nocapture %r) local_unnamed_addr #0 {
entry:
  %wide.load = load <128 x i8>, ptr %a, align 1
  %0 = zext <128 x i8> %wide.load to <128 x i32>
  %1 = add nsw <128 x i32> %0, <i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270>
  store <128 x i32> %1, ptr %r, align 4
  ret void
}

; CHECK-LABEL: test_vadd_const3
; CHECK: [[REG0:(r[0-9]+)]] = #
; CHECK: [[VREG0:(v[0-9]+)]].h = vsplat([[REG0]])
; CHECK: v{{[0-9:]+}}.w = vadd(v{{[0-9]+}}.uh,[[VREG0]].uh)

; Function Attrs: norecurse nounwind
define dso_local void @test_vadd_const3(ptr nocapture readonly %a, ptr nocapture %r) local_unnamed_addr #0 {
entry:
  %wide.load = load <64 x i16>, ptr %a, align 2
  %0 = zext <64 x i16> %wide.load to <64 x i32>
  %1 = add nuw nsw <64 x i32> %0, <i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23>
  store <64 x i32> %1, ptr %r, align 4
  ret void
}

; CHECK-LABEL: test_vadd_const4
; CHECK: [[REG0:(r[0-9]+)]] = #-23
; CHECK: [[VREG0:([0-9]+)]] = vsplat([[REG0]])
; CHECK: [[VREG1:([0-9:]+)]] = v[[VREG0]]
; CHECK: v{{[0-9:]+}}.w = vadd({{.*}}.w,{{v[VREG0]|v[VREG1]}}

; Function Attrs: norecurse nounwind
define dso_local void @test_vadd_const4(ptr nocapture readonly %a, ptr nocapture %r) local_unnamed_addr #0 {
entry:
  %wide.load = load <64 x i16>, ptr %a, align 2
  %0 = zext <64 x i16> %wide.load to <64 x i32>
  %1 = add nsw <64 x i32> %0, <i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23>
  store <64 x i32> %1, ptr %r, align 4
  ret void
}

; CHECK-LABEL: test_vadd_const5
; CHECK: [[REG0:(r[0-9]+)]] = #-257
; CHECK: [[VREG0:([0-9]+)]] = vsplat([[REG0]])
; CHECK: [[VREG1:([0-9:]+)]] = v[[VREG0]]
; CHECK: v{{[0-9:]+}}.w = vadd({{.*}}.w,{{v[VREG0]|v[VREG1]}}

; Function Attrs: norecurse nounwind
define dso_local void @test_vadd_const5(ptr nocapture readonly %a, ptr nocapture %r) local_unnamed_addr #0 {
entry:
  %wide.load = load <64 x i16>, ptr %a, align 2
  %0 = zext <64 x i16> %wide.load to <64 x i32>
  %1 = add nsw <64 x i32> %0, <i32 -257, i32 -257, i32 -257, i32 -257, i32 -257, i32 -257, i32 -257, i32 -257, i32 -257, i32 -257, i32 -257, i32 -257, i32 -257, i32 -257, i32 -257, i32 -257, i32 -257, i32 -257, i32 -257, i32 -257, i32 -257, i32 -257, i32 -257, i32 -257, i32 -257, i32 -257, i32 -257, i32 -257, i32 -257, i32 -257, i32 -257, i32 -257, i32 -257, i32 -257, i32 -257, i32 -257, i32 -257, i32 -257, i32 -257, i32 -257, i32 -257, i32 -257, i32 -257, i32 -257, i32 -257, i32 -257, i32 -257, i32 -257, i32 -257, i32 -257, i32 -257, i32 -257, i32 -257, i32 -257, i32 -257, i32 -257, i32 -257, i32 -257, i32 -257, i32 -257, i32 -257, i32 -257, i32 -257, i32 -257>
  store <64 x i32> %1, ptr %r, align 4
  ret void
}

; CHECK-LABEL: test_vadd_const6
; CHECK: [[REG0:(r[0-9]+)]] = #-23
; CHECK: [[VREG0:(v[0-9]+)]] = vsplat([[REG0]])
; CHECK: v{{[0-9:]+}}.w = vadd({{.*}}[[VREG0]].w{{.*}})

; Function Attrs: norecurse nounwind
define dso_local void @test_vadd_const6(ptr nocapture readonly %a, ptr nocapture %r) local_unnamed_addr #0 {
entry:
  %wide.load = load <32 x i32>, ptr %a, align 4
  %0 = add nsw <32 x i32> %wide.load, <i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23>
  store <32 x i32> %0, ptr %r, align 4
  ret void
}
