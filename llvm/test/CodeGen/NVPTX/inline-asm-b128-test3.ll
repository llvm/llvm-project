; RUN: llc < %s -march=nvptx -mcpu=sm_70 -o - 2>&1  | FileCheck %s

target triple = "nvptx64-nvidia-cuda"

@size = internal addrspace(1) global i32 0, align 4
@value = internal addrspace(1) global i128 0, align 16
@x = internal addrspace(1) global i128 0, align 16
@y = internal addrspace(1) global i128 0, align 16
@z = internal addrspace(1) global i128 0, align 16
@llvm.used = appending global [6 x ptr] [ptr @_Z6kernelv, ptr addrspacecast (ptr addrspace(1) @size to ptr), ptr addrspacecast (ptr addrspace(1) @value to ptr), ptr addrspacecast (ptr addrspace(1) @x to ptr), ptr addrspacecast (ptr addrspace(1) @y to ptr), ptr addrspacecast (ptr addrspace(1) @z to ptr)], section "llvm.metadata"

; Function Attrs: alwaysinline mustprogress willreturn memory(readwrite, argmem: none, inaccessiblemem: none)
define void @_Z6kernelv() {
  ; CHECK-LABEL: _Z6kernelv
  ; CHECK: mov.b128 [[X:%rq[0-9]+]], {{{%rd[0-9]+}}, {{%rd[0-9]+}}};
  ; CHECK: mov.b128 [[Y:%rq[0-9]+]], {{{%rd[0-9]+}}, {{%rd[0-9]+}}};
  ; CHECK: mov.b128 [[Z:%rq[0-9]+]], {{{%rd[0-9]+}}, {{%rd[0-9]+}}};
  ; CHECK: mov.b128 {lo, hi}, [[X]];
  ; CHECK: mov.b128 [[X]], {lo, hi};
  ; CHECK: mov.b128 {lo, hi}, [[Y]];
  ; CHECK: mov.b128 [[Y]], {lo, hi};
  ; CHECK: mov.b128 {lo, hi}, [[Z]];
  ; CHECK: mov.b128 [[Z]], {lo, hi};
  ; CHECK: mov.b128 {[[X_LO:%rd[0-9]+]], [[X_HI:%rd[0-9]+]]}, [[X]];
  ; CHECK: mov.b128 {[[Y_LO:%rd[0-9]+]], [[Y_HI:%rd[0-9]+]]}, [[Y]];
  ; CHECK: mov.b128 {[[Z_LO:%rd[0-9]+]], [[Z_HI:%rd[0-9]+]]}, [[Z]];
  ; CHECK: mov.b128 [[X_NEW:%rq[0-9]+]], {[[X_LO]], [[X_HI]]};
  ; CHECK: mov.b128 [[Y_NEW:%rq[0-9]+]], {[[Y_LO]], [[Y_HI]]};
  ; CHECK: mov.b128 [[Z_NEW:%rq[0-9]+]], {[[Z_LO]], [[Z_HI]]};
  ; CHECK: mov.b128 {lo, hi}, [[X_NEW]];
  ; CHECK: mov.b128 [[X_NEW]], {lo, hi};
  ; CHECK: mov.b128 {lo, hi}, [[Y_NEW]];
  ; CHECK: mov.b128 [[Y_NEW]], {lo, hi};
  ; CHECK: mov.b128 {lo, hi}, [[Z_NEW]];
  ; CHECK: mov.b128 [[Z_NEW]], {lo, hi};
  ; CHECK: mov.b128 {{{%rd[0-9]+}}, {{%rd[0-9]+}}}, [[X_NEW]];
  ; CHECK: mov.b128 {{{%rd[0-9]+}}, {{%rd[0-9]+}}}, [[Y_NEW]];
  ; CHECK: mov.b128 {{{%rd[0-9]+}}, {{%rd[0-9]+}}}, [[Z_NEW]];
  
  %tmp11 = load i32, ptr addrspace(1) @size, align 4
  %cmp3.not = icmp eq i32 %tmp11, 0
  br i1 %cmp3.not, label %._crit_edge, label %.lr.ph.preheader

.lr.ph.preheader:                                 ; preds = %0
  %x.promoted5 = load i128, ptr addrspace(1) @x, align 16
  %y.promoted6 = load i128, ptr addrspace(1) @y, align 16
  %z.promoted7 = load i128, ptr addrspace(1) @z, align 16
  %value.promoted8 = load i128, ptr addrspace(1) @value, align 16
  %umax = sext i32 %tmp11 to i64
  %xtraiter = and i64 %umax, 3
  %1 = icmp ult i32 %tmp11, 4
  br i1 %1, label %._crit_edge.loopexit.unr-lcssa, label %.lr.ph.preheader.new

.lr.ph.preheader.new:                             ; preds = %.lr.ph.preheader
  %unroll_iter = and i64 %umax, -4
  br label %.lr.ph

.lr.ph:                                           ; preds = %.lr.ph, %.lr.ph.preheader.new
  %2 = phi i128 [ %value.promoted8, %.lr.ph.preheader.new ], [ %add14.3, %.lr.ph ]
  %3 = phi i128 [ %z.promoted7, %.lr.ph.preheader.new ], [ %asmresult21.3, %.lr.ph ]
  %4 = phi i128 [ %y.promoted6, %.lr.ph.preheader.new ], [ %asmresult20.3, %.lr.ph ]
  %5 = phi i128 [ %x.promoted5, %.lr.ph.preheader.new ], [ %asmresult19.3, %.lr.ph ]
  %i.04 = phi i64 [ 0, %.lr.ph.preheader.new ], [ %inc.3, %.lr.ph ]
  %niter = phi i64 [ 0, %.lr.ph.preheader.new ], [ %niter.next.3, %.lr.ph ]
  %6 = tail call { i128, i128, i128 } asm "{\0A\09.reg .b64 lo;\0A\09.reg .b64 hi;\0A\09mov.b128 {lo, hi}, $0;\0A\09add.cc.u64 lo, lo, 1;\0A\09mov.b128 $0, {lo, hi};\0A\09mov.b128 {lo, hi}, $1;\0A\09add.cc.u64 hi, hi, 2;\0A\09mov.b128 $1, {lo, hi};\0A\09mov.b128 {lo, hi}, $2;\0A\09add.cc.u64 lo, lo, 3;\0A\09add.cc.u64 hi, hi, 3;\0A\09mov.b128 $2, {lo, hi};\0A\09}\0A\09", "=q,=q,=q,l,0,1,2"(i64 %i.04, i128 %5, i128 %4, i128 %3)
  %asmresult = extractvalue { i128, i128, i128 } %6, 0
  %asmresult7 = extractvalue { i128, i128, i128 } %6, 1
  %asmresult8 = extractvalue { i128, i128, i128 } %6, 2
  %add = add nsw i128 %asmresult, %asmresult7
  %add12 = add nsw i128 %add, %asmresult8
  %add14 = add nsw i128 %add12, %2
  %7 = tail call { i128, i128, i128 } asm "{\0A\09.reg .b64 lo;\0A\09.reg .b64 hi;\0A\09mov.b128 {lo, hi}, $0;\0A\09sub.cc.u64 lo, lo, 1;\0A\09mov.b128 $0, {lo, hi};\0A\09mov.b128 {lo, hi}, $1;\0A\09sub.cc.u64 hi, hi, 2;\0A\09mov.b128 $1, {lo, hi};\0A\09mov.b128 {lo, hi}, $2;\0A\09sub.cc.u64 lo, lo, 3;\0A\09sub.cc.u64 hi, hi, 3;\0A\09mov.b128 $2, {lo, hi};\0A\09}\0A\09", "=q,=q,=q,l,0,1,2"(i64 %i.04, i128 %asmresult, i128 %asmresult7, i128 %asmresult8)
  %asmresult19 = extractvalue { i128, i128, i128 } %7, 0
  %asmresult20 = extractvalue { i128, i128, i128 } %7, 1
  %asmresult21 = extractvalue { i128, i128, i128 } %7, 2
  %inc = add nuw nsw i64 %i.04, 1
  %8 = tail call { i128, i128, i128 } asm "{\0A\09.reg .b64 lo;\0A\09.reg .b64 hi;\0A\09mov.b128 {lo, hi}, $0;\0A\09add.cc.u64 lo, lo, 1;\0A\09mov.b128 $0, {lo, hi};\0A\09mov.b128 {lo, hi}, $1;\0A\09add.cc.u64 hi, hi, 2;\0A\09mov.b128 $1, {lo, hi};\0A\09mov.b128 {lo, hi}, $2;\0A\09add.cc.u64 lo, lo, 3;\0A\09add.cc.u64 hi, hi, 3;\0A\09mov.b128 $2, {lo, hi};\0A\09}\0A\09", "=q,=q,=q,l,0,1,2"(i64 %inc, i128 %asmresult19, i128 %asmresult20, i128 %asmresult21)
  %asmresult.1 = extractvalue { i128, i128, i128 } %8, 0
  %asmresult7.1 = extractvalue { i128, i128, i128 } %8, 1
  %asmresult8.1 = extractvalue { i128, i128, i128 } %8, 2
  %add.1 = add nsw i128 %asmresult.1, %asmresult7.1
  %add12.1 = add nsw i128 %add.1, %asmresult8.1
  %add14.1 = add nsw i128 %add12.1, %add14
  %9 = tail call { i128, i128, i128 } asm "{\0A\09.reg .b64 lo;\0A\09.reg .b64 hi;\0A\09mov.b128 {lo, hi}, $0;\0A\09sub.cc.u64 lo, lo, 1;\0A\09mov.b128 $0, {lo, hi};\0A\09mov.b128 {lo, hi}, $1;\0A\09sub.cc.u64 hi, hi, 2;\0A\09mov.b128 $1, {lo, hi};\0A\09mov.b128 {lo, hi}, $2;\0A\09sub.cc.u64 lo, lo, 3;\0A\09sub.cc.u64 hi, hi, 3;\0A\09mov.b128 $2, {lo, hi};\0A\09}\0A\09", "=q,=q,=q,l,0,1,2"(i64 %inc, i128 %asmresult.1, i128 %asmresult7.1, i128 %asmresult8.1)
  %asmresult19.1 = extractvalue { i128, i128, i128 } %9, 0
  %asmresult20.1 = extractvalue { i128, i128, i128 } %9, 1
  %asmresult21.1 = extractvalue { i128, i128, i128 } %9, 2
  %inc.1 = add nuw nsw i64 %i.04, 2
  %10 = tail call { i128, i128, i128 } asm "{\0A\09.reg .b64 lo;\0A\09.reg .b64 hi;\0A\09mov.b128 {lo, hi}, $0;\0A\09add.cc.u64 lo, lo, 1;\0A\09mov.b128 $0, {lo, hi};\0A\09mov.b128 {lo, hi}, $1;\0A\09add.cc.u64 hi, hi, 2;\0A\09mov.b128 $1, {lo, hi};\0A\09mov.b128 {lo, hi}, $2;\0A\09add.cc.u64 lo, lo, 3;\0A\09add.cc.u64 hi, hi, 3;\0A\09mov.b128 $2, {lo, hi};\0A\09}\0A\09", "=q,=q,=q,l,0,1,2"(i64 %inc.1, i128 %asmresult19.1, i128 %asmresult20.1, i128 %asmresult21.1)
  %asmresult.2 = extractvalue { i128, i128, i128 } %10, 0
  %asmresult7.2 = extractvalue { i128, i128, i128 } %10, 1
  %asmresult8.2 = extractvalue { i128, i128, i128 } %10, 2
  %add.2 = add nsw i128 %asmresult.2, %asmresult7.2
  %add12.2 = add nsw i128 %add.2, %asmresult8.2
  %add14.2 = add nsw i128 %add12.2, %add14.1
  %11 = tail call { i128, i128, i128 } asm "{\0A\09.reg .b64 lo;\0A\09.reg .b64 hi;\0A\09mov.b128 {lo, hi}, $0;\0A\09sub.cc.u64 lo, lo, 1;\0A\09mov.b128 $0, {lo, hi};\0A\09mov.b128 {lo, hi}, $1;\0A\09sub.cc.u64 hi, hi, 2;\0A\09mov.b128 $1, {lo, hi};\0A\09mov.b128 {lo, hi}, $2;\0A\09sub.cc.u64 lo, lo, 3;\0A\09sub.cc.u64 hi, hi, 3;\0A\09mov.b128 $2, {lo, hi};\0A\09}\0A\09", "=q,=q,=q,l,0,1,2"(i64 %inc.1, i128 %asmresult.2, i128 %asmresult7.2, i128 %asmresult8.2)
  %asmresult19.2 = extractvalue { i128, i128, i128 } %11, 0
  %asmresult20.2 = extractvalue { i128, i128, i128 } %11, 1
  %asmresult21.2 = extractvalue { i128, i128, i128 } %11, 2
  %inc.2 = add nuw nsw i64 %i.04, 3
  %12 = tail call { i128, i128, i128 } asm "{\0A\09.reg .b64 lo;\0A\09.reg .b64 hi;\0A\09mov.b128 {lo, hi}, $0;\0A\09add.cc.u64 lo, lo, 1;\0A\09mov.b128 $0, {lo, hi};\0A\09mov.b128 {lo, hi}, $1;\0A\09add.cc.u64 hi, hi, 2;\0A\09mov.b128 $1, {lo, hi};\0A\09mov.b128 {lo, hi}, $2;\0A\09add.cc.u64 lo, lo, 3;\0A\09add.cc.u64 hi, hi, 3;\0A\09mov.b128 $2, {lo, hi};\0A\09}\0A\09", "=q,=q,=q,l,0,1,2"(i64 %inc.2, i128 %asmresult19.2, i128 %asmresult20.2, i128 %asmresult21.2)
  %asmresult.3 = extractvalue { i128, i128, i128 } %12, 0
  %asmresult7.3 = extractvalue { i128, i128, i128 } %12, 1
  %asmresult8.3 = extractvalue { i128, i128, i128 } %12, 2
  %add.3 = add nsw i128 %asmresult.3, %asmresult7.3
  %add12.3 = add nsw i128 %add.3, %asmresult8.3
  %add14.3 = add nsw i128 %add12.3, %add14.2
  %13 = bitcast i128 %add14.3 to <2 x i64>
  store <2 x i64> %13, ptr addrspace(1) @value, align 16
  %14 = tail call { i128, i128, i128 } asm "{\0A\09.reg .b64 lo;\0A\09.reg .b64 hi;\0A\09mov.b128 {lo, hi}, $0;\0A\09sub.cc.u64 lo, lo, 1;\0A\09mov.b128 $0, {lo, hi};\0A\09mov.b128 {lo, hi}, $1;\0A\09sub.cc.u64 hi, hi, 2;\0A\09mov.b128 $1, {lo, hi};\0A\09mov.b128 {lo, hi}, $2;\0A\09sub.cc.u64 lo, lo, 3;\0A\09sub.cc.u64 hi, hi, 3;\0A\09mov.b128 $2, {lo, hi};\0A\09}\0A\09", "=q,=q,=q,l,0,1,2"(i64 %inc.2, i128 %asmresult.3, i128 %asmresult7.3, i128 %asmresult8.3)
  %asmresult19.3 = extractvalue { i128, i128, i128 } %14, 0
  %asmresult20.3 = extractvalue { i128, i128, i128 } %14, 1
  %asmresult21.3 = extractvalue { i128, i128, i128 } %14, 2
  %15 = bitcast i128 %asmresult19.3 to <2 x i64>
  store <2 x i64> %15, ptr addrspace(1) @x, align 16
  %16 = bitcast i128 %asmresult20.3 to <2 x i64>
  store <2 x i64> %16, ptr addrspace(1) @y, align 16
  %17 = bitcast i128 %asmresult21.3 to <2 x i64>
  store <2 x i64> %17, ptr addrspace(1) @z, align 16
  %inc.3 = add nuw i64 %i.04, 4
  %niter.next.3 = add i64 %niter, 4
  %niter.ncmp.3.not = icmp eq i64 %niter.next.3, %unroll_iter
  br i1 %niter.ncmp.3.not, label %._crit_edge.loopexit.unr-lcssa, label %.lr.ph, !llvm.loop !2

._crit_edge.loopexit.unr-lcssa:                   ; preds = %.lr.ph, %.lr.ph.preheader
  %.unr = phi i128 [ %value.promoted8, %.lr.ph.preheader ], [ %add14.3, %.lr.ph ]
  %.unr9 = phi i128 [ %z.promoted7, %.lr.ph.preheader ], [ %asmresult21.3, %.lr.ph ]
  %.unr10 = phi i128 [ %y.promoted6, %.lr.ph.preheader ], [ %asmresult20.3, %.lr.ph ]
  %.unr11 = phi i128 [ %x.promoted5, %.lr.ph.preheader ], [ %asmresult19.3, %.lr.ph ]
  %i.04.unr = phi i64 [ 0, %.lr.ph.preheader ], [ %inc.3, %.lr.ph ]
  %lcmp.mod.not = icmp eq i64 %xtraiter, 0
  br i1 %lcmp.mod.not, label %._crit_edge, label %.lr.ph.epil

.lr.ph.epil:                                      ; preds = %.lr.ph.epil, %._crit_edge.loopexit.unr-lcssa
  %18 = phi i128 [ %add14.epil, %.lr.ph.epil ], [ %.unr, %._crit_edge.loopexit.unr-lcssa ]
  %19 = phi i128 [ %asmresult21.epil, %.lr.ph.epil ], [ %.unr9, %._crit_edge.loopexit.unr-lcssa ]
  %20 = phi i128 [ %asmresult20.epil, %.lr.ph.epil ], [ %.unr10, %._crit_edge.loopexit.unr-lcssa ]
  %21 = phi i128 [ %asmresult19.epil, %.lr.ph.epil ], [ %.unr11, %._crit_edge.loopexit.unr-lcssa ]
  %i.04.epil = phi i64 [ %inc.epil, %.lr.ph.epil ], [ %i.04.unr, %._crit_edge.loopexit.unr-lcssa ]
  %epil.iter = phi i64 [ %epil.iter.next, %.lr.ph.epil ], [ 0, %._crit_edge.loopexit.unr-lcssa ]
  %22 = tail call { i128, i128, i128 } asm "{\0A\09.reg .b64 lo;\0A\09.reg .b64 hi;\0A\09mov.b128 {lo, hi}, $0;\0A\09add.cc.u64 lo, lo, 1;\0A\09mov.b128 $0, {lo, hi};\0A\09mov.b128 {lo, hi}, $1;\0A\09add.cc.u64 hi, hi, 2;\0A\09mov.b128 $1, {lo, hi};\0A\09mov.b128 {lo, hi}, $2;\0A\09add.cc.u64 lo, lo, 3;\0A\09add.cc.u64 hi, hi, 3;\0A\09mov.b128 $2, {lo, hi};\0A\09}\0A\09", "=q,=q,=q,l,0,1,2"(i64 %i.04.epil, i128 %21, i128 %20, i128 %19)
  %asmresult.epil = extractvalue { i128, i128, i128 } %22, 0
  %asmresult7.epil = extractvalue { i128, i128, i128 } %22, 1
  %asmresult8.epil = extractvalue { i128, i128, i128 } %22, 2
  %add.epil = add nsw i128 %asmresult.epil, %asmresult7.epil
  %add12.epil = add nsw i128 %add.epil, %asmresult8.epil
  %add14.epil = add nsw i128 %add12.epil, %18
  %23 = bitcast i128 %add14.epil to <2 x i64>
  store <2 x i64> %23, ptr addrspace(1) @value, align 16
  %24 = tail call { i128, i128, i128 } asm "{\0A\09.reg .b64 lo;\0A\09.reg .b64 hi;\0A\09mov.b128 {lo, hi}, $0;\0A\09sub.cc.u64 lo, lo, 1;\0A\09mov.b128 $0, {lo, hi};\0A\09mov.b128 {lo, hi}, $1;\0A\09sub.cc.u64 hi, hi, 2;\0A\09mov.b128 $1, {lo, hi};\0A\09mov.b128 {lo, hi}, $2;\0A\09sub.cc.u64 lo, lo, 3;\0A\09sub.cc.u64 hi, hi, 3;\0A\09mov.b128 $2, {lo, hi};\0A\09}\0A\09", "=q,=q,=q,l,0,1,2"(i64 %i.04.epil, i128 %asmresult.epil, i128 %asmresult7.epil, i128 %asmresult8.epil)
  %asmresult19.epil = extractvalue { i128, i128, i128 } %24, 0
  %asmresult20.epil = extractvalue { i128, i128, i128 } %24, 1
  %asmresult21.epil = extractvalue { i128, i128, i128 } %24, 2
  %25 = bitcast i128 %asmresult19.epil to <2 x i64>
  store <2 x i64> %25, ptr addrspace(1) @x, align 16
  %26 = bitcast i128 %asmresult20.epil to <2 x i64>
  store <2 x i64> %26, ptr addrspace(1) @y, align 16
  %27 = bitcast i128 %asmresult21.epil to <2 x i64>
  store <2 x i64> %27, ptr addrspace(1) @z, align 16
  %inc.epil = add nuw i64 %i.04.epil, 1
  %epil.iter.next = add i64 %epil.iter, 1
  %epil.iter.cmp.not = icmp eq i64 %epil.iter.next, %xtraiter
  br i1 %epil.iter.cmp.not, label %._crit_edge, label %.lr.ph.epil, !llvm.loop !4

._crit_edge:                                      ; preds = %.lr.ph.epil, %._crit_edge.loopexit.unr-lcssa, %0
  ret void
}


!nvvmir.version = !{!0, !1, !0, !1, !1, !0, !0, !0, !1}

!0 = !{i32 2, i32 0, i32 3, i32 1}
!1 = !{i32 2, i32 0}
!2 = distinct !{!2, !3}
!3 = !{!"llvm.loop.mustprogress"}
!4 = distinct !{!4, !5}
!5 = !{!"llvm.loop.unroll.disable"}
