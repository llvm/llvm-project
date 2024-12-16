; RUN: opt -passes=debugify,loop-vectorize \
; RUN: -force-tail-folding-style=data-with-evl \
; RUN: -prefer-predicate-over-epilogue=predicate-dont-vectorize \
; RUN: -mtriple=riscv64 -mattr=+v -riscv-v-vector-bits-max=128 -S < %s 2>&1 | FileCheck --check-prefix=DEBUGLOC %s

; Testing the debug locations of the generated vector intstruction are same as
; their scalar instruction.

; DEBUGLOC-LABEL: define void @vp_select(
define void @vp_select(ptr noalias %a, ptr noalias %b, ptr noalias %c, i64 %N) {
; DEBUGLOC: vector.body:
; DEBUGLOC:   %[[VPSel:[0-9]+]] = call <vscale x 4 x i32> @llvm.vp.select.nxv4i32(<vscale x 4 x i1> %15, <vscale x 4 x i32> %vp.op.load1, <vscale x 4 x i32> %vp.op, i32 %9)
; DEBUGLOC: for.body:
; DEBUGLOC:   %cond.p = select i1 %cmp4, i32 %22, i32 %23, !dbg !39
 entry:
   br label %for.body

 for.body:
   %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
   %arrayidx = getelementptr inbounds i32, ptr %b, i64 %indvars.iv
   %0 = load i32, ptr %arrayidx, align 4
   %arrayidx3 = getelementptr inbounds i32, ptr %c, i64 %indvars.iv
   %1 = load i32, ptr %arrayidx3, align 4
   %cmp4 = icmp sgt i32 %0, %1
   %2 = sub i32 0, %1
   %cond.p = select i1 %cmp4, i32 %1, i32 %2
   %cond = add i32 %cond.p, %0
   %arrayidx15 = getelementptr inbounds i32, ptr %a, i64 %indvars.iv
   store i32 %cond, ptr %arrayidx15, align 4
   %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
   %exitcond.not = icmp eq i64 %indvars.iv.next, %N
   br i1 %exitcond.not, label %exit, label %for.body

 exit:
   ret void
 }
