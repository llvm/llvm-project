; RUN: llc < %s -march=nvptx -mcpu=sm_70 -mattr=+ptx83 | FileCheck %s
; RUN: %if ptxas %{ llc < %s -march=nvptx -mcpu=sm_70 -mattr=+ptx83 | %ptxas-verify -arch=sm_70 %}

target triple = "nvptx64-nvidia-cuda"

@size = internal addrspace(1) global i32 0, align 4
@x = internal addrspace(1) global i128 0, align 16

define void @test_b128_in_loop() {
  ; CHECK-LABEL: test_b128_in_loop
  ; CHECK: ld.global.u64 [[REG_HI:%rd[0-9]+]], [x+8];
  ; CHECK: ld.global.u64 [[REG_LO:%rd[0-9]+]], [x];
  ; CHECK: mov.b128 [[REG_128:%rq[0-9]+]], {[[REG_LO]], [[REG_HI]]};
  ; CHECK: mov.b128 {lo, hi}, [[REG_128]];
  ; CHECK: add.cc.u64 lo, lo, {{%rd[0-9]+}};
  ; CHECK: mov.b128 [[REG_128]], {lo, hi};

  %tmp11 = load i32, ptr addrspace(1) @size, align 4
  %cmp3.not = icmp eq i32 %tmp11, 0
  br i1 %cmp3.not, label %._crit_edge, label %.lr.ph.preheader

.lr.ph.preheader:                                 ; preds = %0
  %x.promoted5 = load i128, ptr addrspace(1) @x, align 16
  %umax = sext i32 %tmp11 to i64
  br label %.lr.ph

.lr.ph:                                           ; preds = %.lr.ph, %.lr.ph.preheader
  %1 = phi i128 [ %2, %.lr.ph ], [ %x.promoted5, %.lr.ph.preheader ]
  %i.04 = phi i64 [ %inc, %.lr.ph ], [ 0, %.lr.ph.preheader ]
  %2 = tail call i128 asm "{\0A\09.reg .b64 lo;\0A\09.reg .b64 hi;\0A\09mov.b128 {lo, hi}, $0;\0A\09add.cc.u64 lo, lo, $1;\0A\09mov.b128 $0, {lo, hi};\0A\09} \0A\09", "=q,l,0"(i64 %i.04, i128 %1)
  %3 = bitcast i128 %2 to <2 x i64>
  store <2 x i64> %3, ptr addrspace(1) @x, align 16
  %inc = add nuw i64 %i.04, 1
  %exitcond.not = icmp eq i64 %inc, %umax
  br i1 %exitcond.not, label %._crit_edge, label %.lr.ph

._crit_edge:                                      ; preds = %.lr.ph, %0
  ret void
}

!nvvmir.version = !{!0, !1, !0, !1, !1, !0, !0, !0, !1}

!0 = !{i32 2, i32 0, i32 3, i32 1}
!1 = !{i32 2, i32 0}
