; RUN: llc -march=hexagon -mattr=+hvxv73,+hvx-length128b < %s | FileCheck %s

; Make sure that the appropriate vmpy instructions are generated when
; multiplied with a vector of constant values.

; CHECK-LABEL: test_vmpy_const1
; CHECK: v{{[0-9:]+}}.uh = vmpy(v{{[0-9]+}}.ub,r{{[0-9]+}}.ub)
; CHECK: v{{[0-9:]+}}.uw = vunpack(v{{[0-9]+}}.uh)

; Function Attrs: norecurse nounwind
define dso_local void @test_vmpy_const1(ptr nocapture readonly %a, ptr nocapture %r) local_unnamed_addr {
entry:
  %wide.load = load <128 x i8>, ptr %a, align 1
  %0 = zext <128 x i8> %wide.load to <128 x i32>
  %1 = mul nuw nsw <128 x i32> %0, <i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23>
  store <128 x i32> %1, ptr %r, align 4
  ret void
}

; CHECK-LABEL: test_vmpy_const2
; CHECK: v{{[0-9:]+}}.h = vmpy(v{{[0-9]+}}.ub,r{{[0-9]+}}.b)
; CHECK: v{{[0-9:]+}}.w = vunpack(v{{[0-9]+}}.h)

; Function Attrs: norecurse nounwind
define dso_local void @test_vmpy_const2(ptr nocapture readonly %a, ptr nocapture %r) local_unnamed_addr {
entry:
  %wide.load = load <128 x i8>, ptr %a, align 1
  %0 = zext <128 x i8> %wide.load to <128 x i32>
  %1 = mul nsw <128 x i32> %0, <i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23>
  store <128 x i32> %1, ptr %r, align 4
  ret void
}

; CHECK-LABEL: test_vmpy_const2_1
; CHECK: [[REG0:(r[0-9]+)]] = ##-
; CHECK: [[VREG0:(v[0-9]+)]] = vmem
; CHECK: [[VREG1:(v[0-9]+)]] = vsplat([[REG0]])
; CHECK: = vunpack([[VREG0]].ub)
; CHECK: v{{[0-9:]+}}.w = vmpy([[VREG1]].h,v{{[0-9]+}}.uh)

; Function Attrs: norecurse nounwind
define dso_local void @test_vmpy_const2_1(ptr nocapture readonly %a, ptr nocapture %r) local_unnamed_addr {
entry:
  %wide.load = load <128 x i8>, ptr %a, align 1
  %0 = zext <128 x i8> %wide.load to <128 x i32>
  %1 = mul nsw <128 x i32> %0, <i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270, i32 -270>
  store <128 x i32> %1, ptr %r, align 4
  ret void
}

; CHECK-LABEL: test_vmpy_const3
; CHECK: v{{[0-9:]+}}.uw = vmpy(v{{[0-9]+}}.uh,r{{[0-9]+}}.uh)

; Function Attrs: norecurse nounwind
define dso_local void @test_vmpy_const3(ptr nocapture readonly %a, ptr nocapture %r) local_unnamed_addr {
entry:
  %wide.load = load <64 x i16>, ptr %a, align 2
  %0 = zext <64 x i16> %wide.load to <64 x i32>
  %1 = mul nuw nsw <64 x i32> %0, <i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23>
  store <64 x i32> %1, ptr %r, align 4
  ret void
}

; CHECK-LABEL: test_vmpy_const4
; CHECK: [[REG0:(r[0-9]+)]] = #-
; CHECK: [[VREG0:(v[0-9]+)]].h = vsplat([[REG0]])
; CHECK: v{{[0-9:]+}}.w = vmpy([[VREG0]].h,v{{[0-9]+}}.uh)

; Function Attrs: norecurse nounwind
define dso_local void @test_vmpy_const4(ptr nocapture readonly %a, ptr nocapture %r) local_unnamed_addr {
entry:
  %wide.load = load <64 x i16>, ptr %a, align 2
  %0 = zext <64 x i16> %wide.load to <64 x i32>
  %1 = mul nsw <64 x i32> %0, <i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23>
  store <64 x i32> %1, ptr %r, align 4
  ret void
}

; CHECK-LABEL: test_vmpy_const5
; CHECK: [[REG0:(r[0-9]+)]] = #-
; CHECK: [[VREG0:(v[0-9]+)]].h = vsplat([[REG0]])
; CHECK: v{{[0-9:]+}}.w = vmpy([[VREG0]].h,v{{[0-9]+}}.uh)

; Function Attrs: norecurse nounwind
define dso_local void @test_vmpy_const5(ptr nocapture readonly %a, ptr nocapture %r) local_unnamed_addr {
entry:
  %wide.load = load <64 x i16>, ptr %a, align 2
  %0 = zext <64 x i16> %wide.load to <64 x i32>
  %1 = mul nsw <64 x i32> %0, <i32 -257, i32 -257, i32 -257, i32 -257, i32 -257, i32 -257, i32 -257, i32 -257, i32 -257, i32 -257, i32 -257, i32 -257, i32 -257, i32 -257, i32 -257, i32 -257, i32 -257, i32 -257, i32 -257, i32 -257, i32 -257, i32 -257, i32 -257, i32 -257, i32 -257, i32 -257, i32 -257, i32 -257, i32 -257, i32 -257, i32 -257, i32 -257, i32 -257, i32 -257, i32 -257, i32 -257, i32 -257, i32 -257, i32 -257, i32 -257, i32 -257, i32 -257, i32 -257, i32 -257, i32 -257, i32 -257, i32 -257, i32 -257, i32 -257, i32 -257, i32 -257, i32 -257, i32 -257, i32 -257, i32 -257, i32 -257, i32 -257, i32 -257, i32 -257, i32 -257, i32 -257, i32 -257, i32 -257, i32 -257>
  store <64 x i32> %1, ptr %r, align 4
  ret void
}

; CHECK-LABEL: test_vmpy_const6
; CHECK: [[REG0:(r[0-9]+)]] = #-23
; CHECK: [[VREG0:(v[0-9]+)]] = vsplat([[REG0]])
; CHECK: [[VREG1:(v[0-9:]+.w)]] = vmpyieo(v{{[0-9]+}}.h,[[VREG0]].h)
; CHECK: [[VREG1]] += vmpyie

; Function Attrs: norecurse nounwind
define dso_local void @test_vmpy_const6(ptr nocapture readonly %a, ptr nocapture %r) local_unnamed_addr {
entry:
  %wide.load = load <32 x i32>, ptr %a, align 4
  %0 = mul nsw <32 x i32> %wide.load, <i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23, i32 -23>
  store <32 x i32> %0, ptr %r, align 4
  ret void
}

; CHECK-LABEL: test_vmpy_const7
; CHECK: [[REG0:(r[0-9]+)]] = ##.L
; CHECK: [[VREG0:(v[0-9]+)]] = vmemu(r0+#0)
; CHECK: [[VREG1:(v[0-9]+)]] = vmem([[REG0]]+#0)
; CHECK: v{{[0-9:]+}}.h = vmpy([[VREG0]].ub,[[VREG1]].b)

; Function Attrs: norecurse nounwind
define dso_local void @test_vmpy_const7(ptr nocapture readonly %a, ptr nocapture %r) local_unnamed_addr {
entry:
  %wide.load = load <128 x i8>, ptr %a, align 1
  %0 = zext <128 x i8> %wide.load to <128 x i32>
  %1 = mul nsw <128 x i32> %0, <i32 -50, i32 -49, i32 -48, i32 -47, i32 -46, i32 -45, i32 -44, i32 -43, i32 -42, i32 -41, i32 -40, i32 -39, i32 -38, i32 -37, i32 -36, i32 -35, i32 -34, i32 -33, i32 -32, i32 -31, i32 -30, i32 -29, i32 -28, i32 -27, i32 -26, i32 -25, i32 -24, i32 -23, i32 -22, i32 -21, i32 -20, i32 -19, i32 -18, i32 -17, i32 -16, i32 -15, i32 -14, i32 -13, i32 -12, i32 -11, i32 -10, i32 -9, i32 -8, i32 -7, i32 -6, i32 -5, i32 -4, i32 -3, i32 -2, i32 -1, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31, i32 32, i32 33, i32 34, i32 35, i32 36, i32 37, i32 38, i32 39, i32 40, i32 41, i32 42, i32 43, i32 44, i32 45, i32 46, i32 47, i32 48, i32 49, i32 50, i32 51, i32 52, i32 53, i32 54, i32 55, i32 56, i32 57, i32 58, i32 59, i32 60, i32 61, i32 62, i32 63, i32 64, i32 65, i32 66, i32 67, i32 68, i32 69, i32 70, i32 71, i32 72, i32 73, i32 74, i32 75, i32 76, i32 77>
  store <128 x i32> %1, ptr %r, align 4
  ret void
}

; CHECK-LABEL: test_vmpy_const8
; CHECK: v{{[0-9:]+}}.uh = vmpy(v{{[0-9]+}}.ub,r{{[0-9]+}}.ub)

; Function Attrs: norecurse nounwind
define dso_local void @test_vmpy_const8(ptr nocapture readonly %a, ptr nocapture %r) local_unnamed_addr {
entry:
  %wide.load = load <128 x i8>, ptr %a, align 1
  %0 = zext <128 x i8> %wide.load to <128 x i16>
  %1 = mul nuw nsw <128 x i16> %0, <i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20>
  store <128 x i16> %1, ptr %r, align 4
  ret void
}

; CHECK-LABEL: test_vmpy_const9
; CHECK: v{{[0-9:]+}}.h = vmpy(v{{[0-9]+}}.ub,r{{[0-9]+}}.b)

; Function Attrs: norecurse nounwind
define dso_local void @test_vmpy_const9(ptr nocapture readonly %a, ptr nocapture %r) local_unnamed_addr {
entry:
  %wide.load = load <128 x i8>, ptr %a, align 1
  %0 = zext <128 x i8> %wide.load to <128 x i16>
  %1 = mul nuw nsw <128 x i16> %0, <i16 -20, i16 -20, i16 -20, i16 -20, i16 -20, i16 -20, i16 -20, i16 -20, i16 -20, i16 -20, i16 -20, i16 -20, i16 -20, i16 -20, i16 -20, i16 -20, i16 -20, i16 -20, i16 -20, i16 -20, i16 -20, i16 -20, i16 -20, i16 -20, i16 -20, i16 -20, i16 -20, i16 -20, i16 -20, i16 -20, i16 -20, i16 -20, i16 -20, i16 -20, i16 -20, i16 -20, i16 -20, i16 -20, i16 -20, i16 -20, i16 -20, i16 -20, i16 -20, i16 -20, i16 -20, i16 -20, i16 -20, i16 -20, i16 -20, i16 -20, i16 -20, i16 -20, i16 -20, i16 -20, i16 -20, i16 -20, i16 -20, i16 -20, i16 -20, i16 -20, i16 -20, i16 -20, i16 -20, i16 -20, i16 -20, i16 -20, i16 -20, i16 -20, i16 -20, i16 -20, i16 -20, i16 -20, i16 -20, i16 -20, i16 -20, i16 -20, i16 -20, i16 -20, i16 -20, i16 -20, i16 -20, i16 -20, i16 -20, i16 -20, i16 -20, i16 -20, i16 -20, i16 -20, i16 -20, i16 -20, i16 -20, i16 -20, i16 -20, i16 -20, i16 -20, i16 -20, i16 -20, i16 -20, i16 -20, i16 -20, i16 -20, i16 -20, i16 -20, i16 -20, i16 -20, i16 -20, i16 -20, i16 -20, i16 -20, i16 -20, i16 -20, i16 -20, i16 -20, i16 -20, i16 -20, i16 -20, i16 -20, i16 -20, i16 -20, i16 -20, i16 -20, i16 -20, i16 -20, i16 -20, i16 -20, i16 -20, i16 -20, i16 -20>
  store <128 x i16> %1, ptr %r, align 4
  ret void
}

; CHECK-LABEL: test_vmpy_const10
; CHECK: v{{[0-9:]+}}.uw = vmpy(v{{[0-9]+}}.uh,r{{[0-9]+}}.uh)

; Function Attrs: norecurse nounwind
define dso_local void @test_vmpy_const10(ptr nocapture readonly %a, ptr nocapture %r) local_unnamed_addr {
entry:
  %wide.load = load <128 x i16>, ptr %a, align 1
  %0 = zext <128 x i16> %wide.load to <128 x i32>
  %1 = mul nuw nsw <128 x i32> %0, <i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23>
  store <128 x i32> %1, ptr %r, align 4
  ret void
}

; CHECK-LABEL: test_vmpy_const11
; CHECK: v{{[0-9:]+}}.w = vmpy(v{{[0-9]+}}.h,r{{[0-9]+}}.h)

; Function Attrs: norecurse nounwind
define dso_local void @test_vmpy_const11(ptr nocapture readonly %a, ptr nocapture %r) local_unnamed_addr {
entry:
  %wide.load = load <128 x i16>, ptr %a, align 1
  %0 = sext <128 x i16> %wide.load to <128 x i32>
  %1 = mul nuw nsw <128 x i32> %0, <i32 -20, i32 -20, i32 -20, i32 -20, i32 -20, i32 -20, i32 -20, i32 -20, i32 -20, i32 -20, i32 -20, i32 -20, i32 -20, i32 -20, i32 -20, i32 -20, i32 -20, i32 -20, i32 -20, i32 -20, i32 -20, i32 -20, i32 -20, i32 -20, i32 -20, i32 -20, i32 -20, i32 -20, i32 -20, i32 -20, i32 -20, i32 -20, i32 -20, i32 -20, i32 -20, i32 -20, i32 -20, i32 -20, i32 -20, i32 -20, i32 -20, i32 -20, i32 -20, i32 -20, i32 -20, i32 -20, i32 -20, i32 -20, i32 -20, i32 -20, i32 -20, i32 -20, i32 -20, i32 -20, i32 -20, i32 -20, i32 -20, i32 -20, i32 -20, i32 -20, i32 -20, i32 -20, i32 -20, i32 -20, i32 -20, i32 -20, i32 -20, i32 -20, i32 -20, i32 -20, i32 -20, i32 -20, i32 -20, i32 -20, i32 -20, i32 -20, i32 -20, i32 -20, i32 -20, i32 -20, i32 -20, i32 -20, i32 -20, i32 -20, i32 -20, i32 -20, i32 -20, i32 -20, i32 -20, i32 -20, i32 -20, i32 -20, i32 -20, i32 -20, i32 -20, i32 -20, i32 -20, i32 -20, i32 -20, i32 -20, i32 -20, i32 -20, i32 -20, i32 -20, i32 -20, i32 -20, i32 -20, i32 -20, i32 -20, i32 -20, i32 -20, i32 -20, i32 -20, i32 -20, i32 -20, i32 -20, i32 -20, i32 -20, i32 -20, i32 -20, i32 -20, i32 -20, i32 -20, i32 -20, i32 -20, i32 -20, i32 -20, i32 -20>
  store <128 x i32> %1, ptr %r, align 4
  ret void
}

; CHECK-LABEL: test_vmpy_const12
; CHECK: [[VREG0:(v[0-9]+)]] = vmemu(r{{[0-9\+\#0-9]+}})
; CHECK: v{{[0-9:]+}}.h = vmpy(v{{[0-9]+}}.ub,[[VREG0]].b)

; Function Attrs: norecurse nounwind
define dso_local void @test_vmpy_const12(ptr nocapture readonly %a, ptr nocapture %r) local_unnamed_addr {
entry:
  %wide.load = load <128 x i8>, ptr %a, align 1
  %0 = sext <128 x i8> %wide.load to <128 x i16>
  %1 = mul nuw nsw <128 x i16> %0, <i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20>
  store <128 x i16> %1, ptr %r, align 4
  ret void
}

; CHECK-LABEL: test_vmpy_const13
; CHECK: [[VREG0:(v[0-9]+)]] = vmemu(r{{[0-9\+\#0-9]+}})
; CHECK: v{{[0-9:]+}}.w = vmpy([[VREG0]].h,v{{[0-9]+}}.uh)

; Function Attrs: norecurse nounwind
define dso_local void @test_vmpy_const13(ptr nocapture readonly %a, ptr nocapture %r) local_unnamed_addr {
entry:
  %wide.load = load <128 x i16>, ptr %a, align 1
  %0 = sext <128 x i16> %wide.load to <128 x i32>
  %1 = mul nuw nsw <128 x i32> %0, <i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23>
  store <128 x i32> %1, ptr %r, align 4
  ret void
}

; CHECK-LABEL: test_vmpy_const14
; CHECK: v{{[0-9:]+}}.uh = vmpy(v{{[0-9]+}}.ub,r{{[0-9]+}}.ub)

; Function Attrs: norecurse nounwind
define dso_local void @test_vmpy_const14(ptr nocapture readonly %a, ptr nocapture %r) local_unnamed_addr {
entry:
  %wide.load = load <128 x i8>, ptr %a, align 1
  %0 = zext <128 x i8> %wide.load to <128 x i16>
  %1 = shl nuw nsw <128 x i16> %0, <i16 6, i16 6, i16 6, i16 6, i16 6, i16 6, i16 6, i16 6, i16 6, i16 6, i16 6, i16 6, i16 6, i16 6, i16 6, i16 6, i16 6, i16 6, i16 6, i16 6, i16 6, i16 6, i16 6, i16 6, i16 6, i16 6, i16 6, i16 6, i16 6, i16 6, i16 6, i16 6, i16 6, i16 6, i16 6, i16 6, i16 6, i16 6, i16 6, i16 6, i16 6, i16 6, i16 6, i16 6, i16 6, i16 6, i16 6, i16 6, i16 6, i16 6, i16 6, i16 6, i16 6, i16 6, i16 6, i16 6, i16 6, i16 6, i16 6, i16 6, i16 6, i16 6, i16 6, i16 6, i16 6, i16 6, i16 6, i16 6, i16 6, i16 6, i16 6, i16 6, i16 6, i16 6, i16 6, i16 6, i16 6, i16 6, i16 6, i16 6, i16 6, i16 6, i16 6, i16 6, i16 6, i16 6, i16 6, i16 6, i16 6, i16 6, i16 6, i16 6, i16 6, i16 6, i16 6, i16 6, i16 6, i16 6, i16 6, i16 6, i16 6, i16 6, i16 6, i16 6, i16 6, i16 6, i16 6, i16 6, i16 6, i16 6, i16 6, i16 6, i16 6, i16 6, i16 6, i16 6, i16 6, i16 6, i16 6, i16 6, i16 6, i16 6, i16 6, i16 6, i16 6, i16 6, i16 6, i16 6>
  store <128 x i16> %1, ptr %r, align 4
  ret void
}

; CHECK-LABEL: test_vmpy_const15
; CHECK: v{{[0-9:]+}}.uh = vunpack(v{{[0-9]+}}.ub)
; CHECK: v{{[0-9:]+}}.h = vasl(v{{[0-9]+}}.h,r{{[0-9]+}})

; Function Attrs: norecurse nounwind
define dso_local void @test_vmpy_const15(ptr nocapture readonly %a, ptr nocapture %r) local_unnamed_addr {
entry:
  %wide.load = load <128 x i8>, ptr %a, align 1
  %0 = zext <128 x i8> %wide.load to <128 x i16>
  %1 = shl nuw nsw <128 x i16> %0, <i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8>
  store <128 x i16> %1, ptr %r, align 4
  ret void
}

; CHECK-LABEL: test_vmpy_const16
; CHECK: v{{[0-9:]+}}.uw = vmpy(v{{[0-9]+}}.uh,r{{[0-9]+}}.uh)

; Function Attrs: norecurse nounwind
define dso_local void @test_vmpy_const16(ptr nocapture readonly %a, ptr nocapture %r) local_unnamed_addr {
entry:
  %wide.load = load <128 x i16>, ptr %a, align 1
  %0 = zext <128 x i16> %wide.load to <128 x i32>
  %1 = shl nuw nsw <128 x i32> %0, <i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15>
  store <128 x i32> %1, ptr %r, align 4
  ret void
}

; CHECK-LABEL: test_vmpy_const17
; CHECK: v{{[0-9:]+}}.uw = vunpack(v{{[0-9]+}}.uh)
; CHECK: v{{[0-9:]+}}.w = vasl(v{{[0-9]+}}.w,r{{[0-9]+}})

; Function Attrs: norecurse nounwind
define dso_local void @test_vmpy_const17(ptr nocapture readonly %a, ptr nocapture %r) local_unnamed_addr {
entry:
  %wide.load = load <128 x i16>, ptr %a, align 1
  %0 = zext <128 x i16> %wide.load to <128 x i32>
  %1 = shl nuw nsw <128 x i32> %0, <i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16>
  store <128 x i32> %1, ptr %r, align 4
  ret void
}


; CHECK-LABEL: test_vmpy_const18
; CHECK: r{{[0-9]+}} = #2
; CHECK: v{{[0-9:]+}}.b = vsplat(r{{[0-9]+}})
; CHECK: v{{[0-9:]+}}.h = vmpy(v{{[0-9]+}}.ub,v{{[0-9]+}}.b)

; Function Attrs: norecurse nounwind
define dso_local void @test_vmpy_const18(ptr nocapture readonly %a, ptr nocapture %r) local_unnamed_addr {
entry:
  %wide.load = load <128 x i8>, ptr %a, align 1
  %0 = sext <128 x i8> %wide.load to <128 x i32>
  %1 = shl nsw <128 x i32> %0, <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>
  store <128 x i32> %1, ptr %r, align 4
  ret void
}
