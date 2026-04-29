; RUN: llc -mtriple=hexagon -mcpu=hexagonv68 -mattr=+hvxv68,+hvx-length128b \
; RUN:   -pass-remarks-analysis=hexagon-hvx-save %s -o /dev/null 2>&1 \
; RUN:   | FileCheck %s

;; Test that the HVX save remark pass reports caller-save costs around calls.
;; All HVX registers are caller-saved, so any HVX value live across a call
;; requires a save/restore pair on the stack.  The default threshold is 8
;; registers, so we need at least 8 HVX vectors live across the call.

; CHECK: remark: {{.*}} 8 HVX caller-saved register(s) (1024 bytes) live across call
; CHECK-NOT: remark:

declare void @bar()

;; 8 HVX vectors live across a call (8 x 128 = 1024 bytes) -- meets threshold.
define void @test_hvx_save_around_call(ptr %p0, ptr %p1, ptr %p2, ptr %p3,
                                       ptr %p4, ptr %p5, ptr %p6, ptr %p7) {
entry:
  %v0 = load <32 x i32>, ptr %p0, align 128
  %v1 = load <32 x i32>, ptr %p1, align 128
  %v2 = load <32 x i32>, ptr %p2, align 128
  %v3 = load <32 x i32>, ptr %p3, align 128
  %v4 = load <32 x i32>, ptr %p4, align 128
  %v5 = load <32 x i32>, ptr %p5, align 128
  %v6 = load <32 x i32>, ptr %p6, align 128
  %v7 = load <32 x i32>, ptr %p7, align 128
  call void @bar()
  store <32 x i32> %v0, ptr %p0, align 128
  store <32 x i32> %v1, ptr %p1, align 128
  store <32 x i32> %v2, ptr %p2, align 128
  store <32 x i32> %v3, ptr %p3, align 128
  store <32 x i32> %v4, ptr %p4, align 128
  store <32 x i32> %v5, ptr %p5, align 128
  store <32 x i32> %v6, ptr %p6, align 128
  store <32 x i32> %v7, ptr %p7, align 128
  ret void
}

;; Single HVX vector live across call (128 bytes) -- below threshold.
define void @test_below_threshold(ptr %p) {
entry:
  %v = load <32 x i32>, ptr %p, align 128
  call void @bar()
  store <32 x i32> %v, ptr %p, align 128
  ret void
}
