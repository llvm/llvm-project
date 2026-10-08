; RUN: llc -mtriple=hexagon -mcpu=hexagonv68 -mattr=+hvxv68,+hvx-length128b \
; RUN:   -pass-remarks-analysis=hexagon-hvx-save %s -o /dev/null 2>&1 \
; RUN:   | FileCheck %s
;
; RUN: llc -mtriple=hexagon -mcpu=hexagonv68 -mattr=+hvxv68,+hvx-length128b \
; RUN:   -pass-remarks-analysis=hexagon-hvx-save \
; RUN:   -hexagon-hvx-save-threshold=256 %s -o /dev/null 2>&1 \
; RUN:   | FileCheck %s --check-prefix=LOW

;; Test that the HVX save remark pass reports HVX registers live across calls.
;; All HVX registers are caller-saved, so any HVX value live across a call
;; requires a save/restore pair on the stack.  The default threshold is 1024
;; bytes (8 x 128-byte vectors).

; CHECK: remark: {{.*}} 8 HVX caller-saved register(s) (1024 bytes) live across call
; CHECK: remark: {{.*}} 8 HVX caller-saved register(s) (1024 bytes) live across call
; CHECK: remark: {{.*}} 8 HVX caller-saved register(s) (1024 bytes) live across call
; CHECK-NOT: remark:

; LOW: remark: {{.*}} 4 HVX caller-saved register(s) (512 bytes) live across call
; LOW: remark: {{.*}} 8 HVX caller-saved register(s) (1024 bytes) live across call
; LOW: remark: {{.*}} 8 HVX caller-saved register(s) (1024 bytes) live across call
; LOW: remark: {{.*}} 8 HVX caller-saved register(s) (1024 bytes) live across call
; LOW-NOT: remark:

declare void @bar()

;; 4 HVX vectors live across a call (4 x 128 = 512 bytes) -- above LOW threshold,
;; but below the default 1024-byte threshold.
define void @test_four_vecs(ptr %p0, ptr %p1, ptr %p2, ptr %p3) {
entry:
  %v0 = load <32 x i32>, ptr %p0, align 128
  %v1 = load <32 x i32>, ptr %p1, align 128
  %v2 = load <32 x i32>, ptr %p2, align 128
  %v3 = load <32 x i32>, ptr %p3, align 128
  call void @bar()
  store <32 x i32> %v0, ptr %p0, align 128
  store <32 x i32> %v1, ptr %p1, align 128
  store <32 x i32> %v2, ptr %p2, align 128
  store <32 x i32> %v3, ptr %p3, align 128
  ret void
}

;; 8 HVX vectors live across a call (8 x 128 = 1024 bytes) -- meets default
;; threshold.
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

;; 8 HVX vectors loaded in entry, call in a separate call_block, stores in
;; exit.  The vectors are live out of entry and must be counted as live at
;; the call even though the call is in a different basic block.
define void @test_cross_block(ptr %p0, ptr %p1, ptr %p2, ptr %p3,
                              ptr %p4, ptr %p5, ptr %p6, ptr %p7,
                              i1 %cond) {
entry:
  %v0 = load <32 x i32>, ptr %p0, align 128
  %v1 = load <32 x i32>, ptr %p1, align 128
  %v2 = load <32 x i32>, ptr %p2, align 128
  %v3 = load <32 x i32>, ptr %p3, align 128
  %v4 = load <32 x i32>, ptr %p4, align 128
  %v5 = load <32 x i32>, ptr %p5, align 128
  %v6 = load <32 x i32>, ptr %p6, align 128
  %v7 = load <32 x i32>, ptr %p7, align 128
  br label %call_block
call_block:
  call void @bar()
  br label %exit
exit:
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

;; 8 HVX vectors live across a call inside a loop body.  This is the
;; canonical high-cost pattern: the vectors are loaded before the loop,
;; the loop body calls bar(), and the vectors are used after the loop.
define void @test_loop_call(ptr %p0, ptr %p1, ptr %p2, ptr %p3,
                            ptr %p4, ptr %p5, ptr %p6, ptr %p7,
                            i32 %n) {
entry:
  %v0 = load <32 x i32>, ptr %p0, align 128
  %v1 = load <32 x i32>, ptr %p1, align 128
  %v2 = load <32 x i32>, ptr %p2, align 128
  %v3 = load <32 x i32>, ptr %p3, align 128
  %v4 = load <32 x i32>, ptr %p4, align 128
  %v5 = load <32 x i32>, ptr %p5, align 128
  %v6 = load <32 x i32>, ptr %p6, align 128
  %v7 = load <32 x i32>, ptr %p7, align 128
  br label %loop
loop:
  %i = phi i32 [ 0, %entry ], [ %i.next, %loop ]
  call void @bar()
  %i.next = add i32 %i, 1
  %done = icmp eq i32 %i.next, %n
  br i1 %done, label %exit, label %loop
exit:
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
