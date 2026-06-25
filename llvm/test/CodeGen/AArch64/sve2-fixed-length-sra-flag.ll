; REQUIRES: asserts
; RUN: llc -debug-only=isel < %s 2>&1 | FileCheck %s

target triple = "aarch64-unknown-linux-gnu"

define void @usra_disjoint_or32xi8(ptr %a, ptr %b) vscale_range(2,0) #0 {
; CHECK-LABEL: usra_disjoint_or32xi8:
; CHECK:    nxv16i8 = USRA_ZZI_B disjoint {{.*}}, {{.*}}, TargetConstant:i32<4>
  %va = load <32 x i8>, ptr %a
  %vb = load <32 x i8>, ptr %b
  %shift = lshr <32 x i8> %vb, splat(i8 4)
  %res = or disjoint <32 x i8> %va, %shift
  store <32 x i8> %res, ptr %a
  ret void
}

define void @ssra_disjoint_or32xi8(ptr %a, ptr %b) vscale_range(2,0) #0 {
; CHECK-LABEL: ssra_disjoint_or32xi8:
; CHECK:    nxv16i8 = SSRA_ZZI_B disjoint {{.*}}, {{.*}}, TargetConstant:i32<4>
  %va = load <32 x i8>, ptr %a
  %vb = load <32 x i8>, ptr %b
  %shift = ashr <32 x i8> %vb, splat(i8 4)
  %res = or disjoint <32 x i8> %va, %shift
  store <32 x i8> %res, ptr %a
  ret void
}

attributes #0 = { "target-features"="+sve,+sve2" }
