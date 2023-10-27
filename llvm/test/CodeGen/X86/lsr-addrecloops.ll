; RUN: llc < %s | FileCheck %s

; Check that the SCEVs produced from the multiple loops don't attempt to get
; combines in invalid ways. The LSR filtering could attempt to combine addrecs
; from different loops.

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @in4dob_(ptr nocapture writeonly %0, ptr nocapture readonly %1, ptr nocapture readonly %2, i64 %3, i1 %min.iters.check840) "target-cpu"="icelake-server" {
; CHECK-LABEL: in4dob_:
; CHECK:       .LBB0_6: # %vector.body807
; CHECK-NEXT:    # =>This Inner Loop Header: Depth=1
; CHECK-NEXT:    leaq (%rdi,%r9), %r11
; CHECK-NEXT:    vmovups %ymm0, (%rax,%r11)
; CHECK-NEXT:    vmovups %ymm0, 1(%rax,%r11)
; CHECK-NEXT:    vmovups %ymm0, 2(%rax,%r11)
; CHECK-NEXT:    vmovups %ymm0, 3(%rax,%r11)
; CHECK-NEXT:    vmovups %ymm0, 4(%rax,%r11)
; CHECK-NEXT:    vmovups %ymm0, 5(%rax,%r11)
; CHECK-NEXT:    vmovups %ymm0, 6(%rax,%r11)
; CHECK-NEXT:    vmovups %ymm0, 7(%rax,%r11)
; CHECK-NEXT:    addq $8, %r9
; CHECK-NEXT:    cmpq %r9, %r10
; CHECK-NEXT:    jne .LBB0_6
; CHECK:       .LBB0_14: # %vector.body847
; CHECK-NEXT:    # =>This Inner Loop Header: Depth=1
; CHECK-NEXT:    leaq (%rdi,%rcx), %r8
; CHECK-NEXT:    vmovups %ymm0, 96(%rax,%r8)
; CHECK-NEXT:    vmovups %ymm0, 97(%rax,%r8)
; CHECK-NEXT:    vmovups %ymm0, 98(%rax,%r8)
; CHECK-NEXT:    vmovups %ymm0, 99(%rax,%r8)
; CHECK-NEXT:    vmovups %ymm0, 100(%rax,%r8)
; CHECK-NEXT:    vmovups %ymm0, 101(%rax,%r8)
; CHECK-NEXT:    vmovups %ymm0, 102(%rax,%r8)
; CHECK-NEXT:    vmovups %ymm0, 103(%rax,%r8)
; CHECK-NEXT:    addq $8, %rcx
; CHECK-NEXT:    cmpq %rcx, %rdx
; CHECK-NEXT:    jne .LBB0_14
.preheader263:
  %4 = shl i64 %3, 2
  br label %5

5:                                                ; preds = %16, %.preheader263
  %lsr.iv1135 = phi ptr [ %0, %.preheader263 ], [ %uglygep1136, %16 ]
  %indvars.iv487 = phi i64 [ 1, %.preheader263 ], [ %indvars.iv.next488, %16 ]
  %6 = getelementptr float, ptr %1, i64 %indvars.iv487
  %7 = load float, ptr %6, align 4
  %8 = fcmp oeq float %7, 0.000000e+00
  %9 = getelementptr float, ptr %2, i64 %indvars.iv487
  %10 = load float, ptr %9, align 4
  %11 = fcmp oeq float %10, 0.000000e+00
  %12 = and i1 %8, %11
  br i1 %12, label %vector.body807.preheader, label %16

vector.body807.preheader:                         ; preds = %5
  %13 = add i64 %3, 1
  %xtraiter = and i64 %13, 7
  %14 = icmp ult i64 %3, 7
  br i1 %14, label %.lr.ph373.unr-lcssa, label %vector.body807.preheader.new

vector.body807.preheader.new:                     ; preds = %vector.body807.preheader
  %unroll_iter = and i64 %13, -8
  br label %vector.body807

vector.body807:                                   ; preds = %vector.body807, %vector.body807.preheader.new
  %lsr.iv1194 = phi i64 [ 0, %vector.body807.preheader.new ], [ %lsr.iv.next1195.7, %vector.body807 ]
  %niter = phi i64 [ 0, %vector.body807.preheader.new ], [ %niter.next.7, %vector.body807 ]
  %uglygep1197 = getelementptr i8, ptr %lsr.iv1135, i64 %lsr.iv1194
  store <8 x float> zeroinitializer, ptr %uglygep1197, align 4
  %lsr.iv.next1195 = or i64 %lsr.iv1194, 1
  %uglygep1197.1 = getelementptr i8, ptr %lsr.iv1135, i64 %lsr.iv.next1195
  store <8 x float> zeroinitializer, ptr %uglygep1197.1, align 4
  %lsr.iv.next1195.1 = or i64 %lsr.iv1194, 2
  %uglygep1197.2 = getelementptr i8, ptr %lsr.iv1135, i64 %lsr.iv.next1195.1
  store <8 x float> zeroinitializer, ptr %uglygep1197.2, align 4
  %lsr.iv.next1195.2 = or i64 %lsr.iv1194, 3
  %uglygep1197.3 = getelementptr i8, ptr %lsr.iv1135, i64 %lsr.iv.next1195.2
  store <8 x float> zeroinitializer, ptr %uglygep1197.3, align 4
  %lsr.iv.next1195.3 = or i64 %lsr.iv1194, 4
  %uglygep1197.4 = getelementptr i8, ptr %lsr.iv1135, i64 %lsr.iv.next1195.3
  store <8 x float> zeroinitializer, ptr %uglygep1197.4, align 4
  %lsr.iv.next1195.4 = or i64 %lsr.iv1194, 5
  %uglygep1197.5 = getelementptr i8, ptr %lsr.iv1135, i64 %lsr.iv.next1195.4
  store <8 x float> zeroinitializer, ptr %uglygep1197.5, align 4
  %lsr.iv.next1195.5 = or i64 %lsr.iv1194, 6
  %uglygep1197.6 = getelementptr i8, ptr %lsr.iv1135, i64 %lsr.iv.next1195.5
  store <8 x float> zeroinitializer, ptr %uglygep1197.6, align 4
  %lsr.iv.next1195.6 = or i64 %lsr.iv1194, 7
  %uglygep1197.7 = getelementptr i8, ptr %lsr.iv1135, i64 %lsr.iv.next1195.6
  store <8 x float> zeroinitializer, ptr %uglygep1197.7, align 4
  %lsr.iv.next1195.7 = add i64 %lsr.iv1194, 8
  %niter.next.7 = add i64 %niter, 8
  %niter.ncmp.7 = icmp eq i64 %niter.next.7, %unroll_iter
  br i1 %niter.ncmp.7, label %.lr.ph373.unr-lcssa.loopexit, label %vector.body807

.lr.ph373.unr-lcssa.loopexit:                     ; preds = %vector.body807
  br label %.lr.ph373.unr-lcssa

.lr.ph373.unr-lcssa:                              ; preds = %.lr.ph373.unr-lcssa.loopexit, %vector.body807.preheader
  %lsr.iv1194.unr = phi i64 [ 0, %vector.body807.preheader ], [ %lsr.iv.next1195.7, %.lr.ph373.unr-lcssa.loopexit ]
  %lcmp.mod.not = icmp eq i64 %xtraiter, 0
  br i1 %lcmp.mod.not, label %.lr.ph373, label %vector.body807.epil.preheader

vector.body807.epil.preheader:                    ; preds = %.lr.ph373.unr-lcssa
  br label %vector.body807.epil

vector.body807.epil:                              ; preds = %vector.body807.epil.preheader, %vector.body807.epil
  %lsr.iv1194.epil = phi i64 [ %lsr.iv.next1195.epil, %vector.body807.epil ], [ %lsr.iv1194.unr, %vector.body807.epil.preheader ]
  %epil.iter = phi i64 [ %epil.iter.next, %vector.body807.epil ], [ 0, %vector.body807.epil.preheader ]
  %uglygep1197.epil = getelementptr i8, ptr %lsr.iv1135, i64 %lsr.iv1194.epil
  store <8 x float> zeroinitializer, ptr %uglygep1197.epil, align 4
  %lsr.iv.next1195.epil = add i64 %lsr.iv1194.epil, 1
  %epil.iter.next = add i64 %epil.iter, 1
  %epil.iter.cmp.not = icmp eq i64 %epil.iter.next, %xtraiter
  br i1 %epil.iter.cmp.not, label %.lr.ph373.loopexit, label %vector.body807.epil

.lr.ph373.loopexit:                               ; preds = %vector.body807.epil
  br label %.lr.ph373

.lr.ph373:                                        ; preds = %.lr.ph373.loopexit, %.lr.ph373.unr-lcssa
  br i1 %min.iters.check840, label %scalar.ph839.preheader, label %vector.body847.preheader

vector.body847.preheader:                         ; preds = %.lr.ph373
  %uglygep11551 = getelementptr i8, ptr %lsr.iv1135, i64 96
  %xtraiter12 = and i64 %13, 7
  %15 = icmp ult i64 %3, 7
  br i1 %15, label %common.ret.loopexit.unr-lcssa, label %vector.body847.preheader.new

vector.body847.preheader.new:                     ; preds = %vector.body847.preheader
  %unroll_iter15 = and i64 %13, -8
  br label %vector.body847

vector.body847:                                   ; preds = %vector.body847, %vector.body847.preheader.new
  %lsr.iv1152 = phi i64 [ 0, %vector.body847.preheader.new ], [ %lsr.iv.next1153.7, %vector.body847 ]
  %niter16 = phi i64 [ 0, %vector.body847.preheader.new ], [ %niter16.next.7, %vector.body847 ]
  %uglygep1156 = getelementptr i8, ptr %uglygep11551, i64 %lsr.iv1152
  store <8 x float> zeroinitializer, ptr %uglygep1156, align 4
  %lsr.iv.next1153 = or i64 %lsr.iv1152, 1
  %uglygep1156.1 = getelementptr i8, ptr %uglygep11551, i64 %lsr.iv.next1153
  store <8 x float> zeroinitializer, ptr %uglygep1156.1, align 4
  %lsr.iv.next1153.1 = or i64 %lsr.iv1152, 2
  %uglygep1156.2 = getelementptr i8, ptr %uglygep11551, i64 %lsr.iv.next1153.1
  store <8 x float> zeroinitializer, ptr %uglygep1156.2, align 4
  %lsr.iv.next1153.2 = or i64 %lsr.iv1152, 3
  %uglygep1156.3 = getelementptr i8, ptr %uglygep11551, i64 %lsr.iv.next1153.2
  store <8 x float> zeroinitializer, ptr %uglygep1156.3, align 4
  %lsr.iv.next1153.3 = or i64 %lsr.iv1152, 4
  %uglygep1156.4 = getelementptr i8, ptr %uglygep11551, i64 %lsr.iv.next1153.3
  store <8 x float> zeroinitializer, ptr %uglygep1156.4, align 4
  %lsr.iv.next1153.4 = or i64 %lsr.iv1152, 5
  %uglygep1156.5 = getelementptr i8, ptr %uglygep11551, i64 %lsr.iv.next1153.4
  store <8 x float> zeroinitializer, ptr %uglygep1156.5, align 4
  %lsr.iv.next1153.5 = or i64 %lsr.iv1152, 6
  %uglygep1156.6 = getelementptr i8, ptr %uglygep11551, i64 %lsr.iv.next1153.5
  store <8 x float> zeroinitializer, ptr %uglygep1156.6, align 4
  %lsr.iv.next1153.6 = or i64 %lsr.iv1152, 7
  %uglygep1156.7 = getelementptr i8, ptr %uglygep11551, i64 %lsr.iv.next1153.6
  store <8 x float> zeroinitializer, ptr %uglygep1156.7, align 4
  %lsr.iv.next1153.7 = add i64 %lsr.iv1152, 8
  %niter16.next.7 = add i64 %niter16, 8
  %niter16.ncmp.7 = icmp eq i64 %niter16.next.7, %unroll_iter15
  br i1 %niter16.ncmp.7, label %common.ret.loopexit.unr-lcssa.loopexit, label %vector.body847

common.ret.loopexit.unr-lcssa.loopexit:           ; preds = %vector.body847
  br label %common.ret.loopexit.unr-lcssa

common.ret.loopexit.unr-lcssa:                    ; preds = %common.ret.loopexit.unr-lcssa.loopexit, %vector.body847.preheader
  %lsr.iv1152.unr = phi i64 [ 0, %vector.body847.preheader ], [ %lsr.iv.next1153.7, %common.ret.loopexit.unr-lcssa.loopexit ]
  %lcmp.mod14.not = icmp eq i64 %xtraiter12, 0
  br i1 %lcmp.mod14.not, label %common.ret, label %vector.body847.epil.preheader

vector.body847.epil.preheader:                    ; preds = %common.ret.loopexit.unr-lcssa
  br label %vector.body847.epil

vector.body847.epil:                              ; preds = %vector.body847.epil.preheader, %vector.body847.epil
  %lsr.iv1152.epil = phi i64 [ %lsr.iv.next1153.epil, %vector.body847.epil ], [ %lsr.iv1152.unr, %vector.body847.epil.preheader ]
  %epil.iter13 = phi i64 [ %epil.iter13.next, %vector.body847.epil ], [ 0, %vector.body847.epil.preheader ]
  %uglygep1156.epil = getelementptr i8, ptr %uglygep11551, i64 %lsr.iv1152.epil
  store <8 x float> zeroinitializer, ptr %uglygep1156.epil, align 4
  %lsr.iv.next1153.epil = add i64 %lsr.iv1152.epil, 1
  %epil.iter13.next = add i64 %epil.iter13, 1
  %epil.iter13.cmp.not = icmp eq i64 %epil.iter13.next, %xtraiter12
  br i1 %epil.iter13.cmp.not, label %common.ret.loopexit, label %vector.body847.epil

common.ret.loopexit:                              ; preds = %vector.body847.epil
  br label %common.ret

common.ret.loopexit1:                             ; preds = %16
  br label %common.ret

common.ret:                                       ; preds = %common.ret.loopexit1, %common.ret.loopexit, %scalar.ph839.preheader, %common.ret.loopexit.unr-lcssa
  ret void

scalar.ph839.preheader:                           ; preds = %.lr.ph373
  store float 0.000000e+00, ptr %0, align 4
  br label %common.ret

16:                                               ; preds = %5
  %indvars.iv.next488 = add i64 %indvars.iv487, 1
  %exitcond492.not = icmp eq i64 %indvars.iv.next488, %3
  %uglygep1136 = getelementptr i8, ptr %lsr.iv1135, i64 %4
  br i1 %exitcond492.not, label %common.ret.loopexit1, label %5
}
