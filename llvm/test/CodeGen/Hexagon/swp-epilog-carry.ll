; RUN: llc -mtriple=hexagon -mcpu=hexagonv68 -O2 < %s | FileCheck %s

; Regression test for https://github.com/llvm/llvm-project/issues/208943.
;
; The pipelined epilog of the unrolled remainder loop must use the carry
; produced by the last kernel iteration, not the initial carry from the
; prolog. The buggy code emits an extra "r4 = r5" copy between :endloop0
; and the epilog packet that clobbers the correct carry.

; CHECK-LABEL: mp_div_2d:

; CHECK-LABEL: %for.body.epil
; CHECK:      loop0(.[[KERNEL:LBB[0-9]+_[0-9]+]],
; CHECK:      .[[KERNEL]]:{{.*}}%for.body.epil
; CHECK:      [[CARRY:r[0-9]+]] |= asl([[PRIOR:r[0-9]+]],r{{[0-9]+}})
; CHECK-NEXT: [[PRIOR]] = r{{[0-9]+}}
; CHECK:      :endloop0
; CHECK-NOT:  [[PRIOR]] = r
; CHECK:      [[CARRY]] |= asl([[PRIOR]],r{{[0-9]+}})
; CHECK-NEXT: memw({{.*}}) = [[CARRY]].new

define dso_local i32 @mp_div_2d(ptr noundef %a, i32 noundef %b, ptr noundef %c, ptr noundef %d) #0 {
entry:
  %cmp = icmp slt i32 %b, 1
  %call = tail call i32 @mp_copy(ptr noundef %a, ptr noundef %c)
  br i1 %cmp, label %if.then, label %if.end3

if.then:
  %cmp1.not = icmp eq ptr %d, null
  br i1 %cmp1.not, label %cleanup, label %if.then2

if.then2:
  tail call void @mp_zero(ptr noundef nonnull %d)
  br label %cleanup

if.end3:
  %cmp5.not = icmp eq i32 %call, 0
  br i1 %cmp5.not, label %if.end7, label %cleanup

if.end7:
  %cmp8.not = icmp eq ptr %d, null
  br i1 %cmp8.not, label %if.end14, label %if.then9

if.then9:
  %call10 = tail call i32 @mp_mod_2d(ptr noundef %a, i32 noundef %b, ptr noundef nonnull %d)
  %cmp11.not = icmp eq i32 %call10, 0
  br i1 %cmp11.not, label %if.end14, label %cleanup

if.end14:
  %cmp15 = icmp samesign ugt i32 %b, 27
  br i1 %cmp15, label %if.then16, label %if.end17

if.then16:
  %div = udiv i32 %b, 28
  tail call void @mp_rshd(ptr noundef %c, i32 noundef %div)
  br label %if.end17

if.end17:
  %rem = urem i32 %b, 28
  %cmp18.not = icmp eq i32 %rem, 0
  br i1 %cmp18.not, label %if.end26, label %if.then19

if.then19:
  %notmask = shl nsw i32 -1, %rem
  %sub = xor i32 %notmask, -1
  %sub20 = sub nuw nsw i32 28, %rem
  %0 = load i32, ptr %c, align 4
  %cmp2455 = icmp sgt i32 %0, 0
  br i1 %cmp2455, label %for.body.preheader, label %if.end26

for.body.preheader:
  %dp = getelementptr inbounds nuw i8, ptr %c, i32 12
  %1 = load ptr, ptr %dp, align 4
  %2 = getelementptr [4 x i8], ptr %1, i32 %0
  %add.ptr = getelementptr i8, ptr %2, i32 -4
  %xtraiter = and i32 %0, 7
  %3 = icmp ult i32 %0, 8
  br i1 %3, label %for.body.epil.preheader, label %for.body.preheader.new

for.body.preheader.new:
  %unroll_iter = and i32 %0, 2147483640
  br label %for.body

for.body:
  %r.058 = phi i32 [ 0, %for.body.preheader.new ], [ %and.7, %for.body ]
  %tmpc.057 = phi ptr [ %add.ptr, %for.body.preheader.new ], [ %incdec.ptr.7, %for.body ]
  %niter = phi i32 [ 0, %for.body.preheader.new ], [ %niter.next.7, %for.body ]
  %4 = load i32, ptr %tmpc.057, align 4
  %shr = lshr i32 %4, %rem
  %shl25 = shl i32 %r.058, %sub20
  %or = or i32 %shr, %shl25
  store i32 %or, ptr %tmpc.057, align 4
  %incdec.ptr = getelementptr inbounds i8, ptr %tmpc.057, i32 -4
  %5 = load i32, ptr %incdec.ptr, align 4
  %shr.1 = lshr i32 %5, %rem
  %6 = shl i32 %4, %sub20
  %shl25.1 = and i32 %6, 268435455
  %or.1 = or i32 %shr.1, %shl25.1
  store i32 %or.1, ptr %incdec.ptr, align 4
  %incdec.ptr.1 = getelementptr inbounds i8, ptr %tmpc.057, i32 -8
  %7 = load i32, ptr %incdec.ptr.1, align 4
  %shr.2 = lshr i32 %7, %rem
  %8 = shl i32 %5, %sub20
  %shl25.2 = and i32 %8, 268435455
  %or.2 = or i32 %shr.2, %shl25.2
  store i32 %or.2, ptr %incdec.ptr.1, align 4
  %incdec.ptr.2 = getelementptr inbounds i8, ptr %tmpc.057, i32 -12
  %9 = load i32, ptr %incdec.ptr.2, align 4
  %shr.3 = lshr i32 %9, %rem
  %10 = shl i32 %7, %sub20
  %shl25.3 = and i32 %10, 268435455
  %or.3 = or i32 %shr.3, %shl25.3
  store i32 %or.3, ptr %incdec.ptr.2, align 4
  %incdec.ptr.3 = getelementptr inbounds i8, ptr %tmpc.057, i32 -16
  %11 = load i32, ptr %incdec.ptr.3, align 4
  %shr.4 = lshr i32 %11, %rem
  %12 = shl i32 %9, %sub20
  %shl25.4 = and i32 %12, 268435455
  %or.4 = or i32 %shr.4, %shl25.4
  store i32 %or.4, ptr %incdec.ptr.3, align 4
  %incdec.ptr.4 = getelementptr inbounds i8, ptr %tmpc.057, i32 -20
  %13 = load i32, ptr %incdec.ptr.4, align 4
  %shr.5 = lshr i32 %13, %rem
  %14 = shl i32 %11, %sub20
  %shl25.5 = and i32 %14, 268435455
  %or.5 = or i32 %shr.5, %shl25.5
  store i32 %or.5, ptr %incdec.ptr.4, align 4
  %incdec.ptr.5 = getelementptr inbounds i8, ptr %tmpc.057, i32 -24
  %15 = load i32, ptr %incdec.ptr.5, align 4
  %shr.6 = lshr i32 %15, %rem
  %16 = shl i32 %13, %sub20
  %shl25.6 = and i32 %16, 268435455
  %or.6 = or i32 %shr.6, %shl25.6
  store i32 %or.6, ptr %incdec.ptr.5, align 4
  %incdec.ptr.6 = getelementptr inbounds i8, ptr %tmpc.057, i32 -28
  %17 = load i32, ptr %incdec.ptr.6, align 4
  %and.7 = and i32 %17, %sub
  %shr.7 = lshr i32 %17, %rem
  %18 = shl i32 %15, %sub20
  %shl25.7 = and i32 %18, 268435455
  %or.7 = or i32 %shr.7, %shl25.7
  store i32 %or.7, ptr %incdec.ptr.6, align 4
  %incdec.ptr.7 = getelementptr inbounds i8, ptr %tmpc.057, i32 -32
  %niter.next.7 = add i32 %niter, 8
  %niter.ncmp.7 = icmp eq i32 %niter.next.7, %unroll_iter
  br i1 %niter.ncmp.7, label %if.end26.loopexit.unr-lcssa, label %for.body

if.end26.loopexit.unr-lcssa:
  %lcmp.mod.not = icmp eq i32 %xtraiter, 0
  br i1 %lcmp.mod.not, label %if.end26, label %for.body.epil.preheader

for.body.epil.preheader:
  %r.058.epil.init = phi i32 [ 0, %for.body.preheader ], [ %and.7, %if.end26.loopexit.unr-lcssa ]
  %tmpc.057.epil.init = phi ptr [ %add.ptr, %for.body.preheader ], [ %incdec.ptr.7, %if.end26.loopexit.unr-lcssa ]
  %lcmp.mod59 = icmp ne i32 %xtraiter, 0
  tail call void @llvm.assume(i1 %lcmp.mod59)
  br label %for.body.epil

for.body.epil:
  %r.058.epil = phi i32 [ %and.epil, %for.body.epil ], [ %r.058.epil.init, %for.body.epil.preheader ]
  %tmpc.057.epil = phi ptr [ %incdec.ptr.epil, %for.body.epil ], [ %tmpc.057.epil.init, %for.body.epil.preheader ]
  %epil.iter = phi i32 [ %epil.iter.next, %for.body.epil ], [ 0, %for.body.epil.preheader ]
  %19 = load i32, ptr %tmpc.057.epil, align 4
  %and.epil = and i32 %19, %sub
  %shr.epil = lshr i32 %19, %rem
  %shl25.epil = shl i32 %r.058.epil, %sub20
  %or.epil = or i32 %shr.epil, %shl25.epil
  store i32 %or.epil, ptr %tmpc.057.epil, align 4
  %incdec.ptr.epil = getelementptr inbounds i8, ptr %tmpc.057.epil, i32 -4
  %epil.iter.next = add i32 %epil.iter, 1
  %epil.iter.cmp.not = icmp eq i32 %epil.iter.next, %xtraiter
  br i1 %epil.iter.cmp.not, label %if.end26, label %for.body.epil

if.end26:
  tail call void @mp_clamp(ptr noundef %c)
  br label %cleanup

cleanup:
  %retval.0 = phi i32 [ 0, %if.end26 ], [ %call, %if.then ], [ %call, %if.end3 ], [ %call, %if.then2 ], [ %call10, %if.then9 ]
  ret i32 %retval.0
}

declare i32 @mp_copy(ptr noundef, ptr noundef)
declare void @mp_zero(ptr noundef)
declare i32 @mp_mod_2d(ptr noundef, i32 noundef, ptr noundef)
declare void @mp_rshd(ptr noundef, i32 noundef)
declare void @mp_clamp(ptr noundef)
declare void @llvm.assume(i1 noundef)

attributes #0 = { nounwind "target-cpu"="hexagonv68" }
