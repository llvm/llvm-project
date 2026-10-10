; RUN: opt -passes=loop-vectorize -force-vector-width=8 -enable-epilogue-vectorization \
; RUN:     -epilogue-vectorization-force-VF=4 -disable-output \
; RUN:     -vplan-print-after=printFinalVPlan -vplan-verify-each %s 2>&1 | FileCheck %s

; Check how plans for epilogue vectorization are represented in VPlan.

define i64 @resume_values(ptr noalias %A, i64 %n) {
; CHECK-LABEL: VPlan for loop in 'resume_values'
; CHECK:  VPlan 'Final VPlan for VF={8},UF={1}' {
; CHECK-NEXT:  Live-in ir<%n> = original trip-count
; CHECK-EMPTY:
; CHECK-NEXT:  ir-bb<entry>:
; CHECK-NEXT:    EMIT vp<%min.iters.check> = icmp ult ir<%n>, ir<4>
; CHECK-NEXT:    EMIT branch-on-cond vp<%min.iters.check>
; CHECK-NEXT:  Successor(s): ir-bb<scalar.ph>, vector.main.loop.iter.check
; CHECK-EMPTY:
; CHECK-NEXT:  vector.main.loop.iter.check:
; CHECK-NEXT:    EMIT vp<%min.iters.check>.1 = icmp ult ir<%n>, ir<8>
; CHECK-NEXT:    EMIT branch-on-cond vp<%min.iters.check>.1
; CHECK-NEXT:  Successor(s): ir-bb<scalar.ph>, vector.ph
; CHECK-EMPTY:
; CHECK-NEXT:  vector.ph:
; CHECK-NEXT:    EMIT vp<[[VP4:%[0-9]+]]> = and ir<%n>, ir<7>
; CHECK-NEXT:    EMIT vp<%n.vec> = sub ir<%n>, vp<[[VP4]]>
; CHECK-NEXT:    EMIT vp<[[VP5:%[0-9]+]]> = reduction-start-vector ir<5>, ir<0>, ir<1>
; CHECK-NEXT:  Successor(s): vector.body
; CHECK-EMPTY:
; CHECK-NEXT:  vector.body:
; CHECK-NEXT:    EMIT-SCALAR vp<%index> = phi [ ir<0>, vector.ph ], [ vp<%index.next>, vector.body ]
; CHECK-NEXT:    WIDEN-REDUCTION-PHI ir<%red> = phi (add) vp<[[VP5]]>, ir<%red.next>
; CHECK-NEXT:    CLONE ir<%gep> = getelementptr inbounds ir<%A>, vp<%index>
; CHECK-NEXT:    WIDEN ir<%l> = load ir<%gep>
; CHECK-NEXT:    WIDEN ir<%red.next> = add ir<%red>, ir<%l>
; CHECK-NEXT:    EMIT vp<%index.next> = add nuw vp<%index>, ir<8>
; CHECK-NEXT:    EMIT vp<[[VP6:%[0-9]+]]> = icmp eq vp<%index.next>, vp<%n.vec>
; CHECK-NEXT:    EMIT branch-on-cond vp<[[VP6]]>
; CHECK-NEXT:  Successor(s): middle.block, vector.body
; CHECK-EMPTY:
; CHECK-NEXT:  middle.block:
; CHECK-NEXT:    EMIT-SCALAR vp<%{{.+}}> = resume-for-epilogue vp<%n.vec>, ir<0>
; CHECK-NEXT:    EMIT vp<[[VP9:%[0-9]+]]> = compute-reduction-result (add) ir<%red.next>
; CHECK-NEXT:    EMIT vp<%cmp.n> = icmp eq ir<%n>, vp<%n.vec>
; CHECK-NEXT:    EMIT-SCALAR vp<%{{.+}}> = resume-for-epilogue vp<%n.vec>, ir<0>
; CHECK-NEXT:    EMIT-SCALAR vp<%{{.+}}> = resume-for-epilogue vp<[[VP9]]>, ir<5>
; CHECK-NEXT:    EMIT branch-on-cond vp<%cmp.n>
; CHECK-NEXT:  Successor(s): ir-bb<exit>, ir-bb<scalar.ph>
; CHECK-EMPTY:
; CHECK-NEXT:  ir-bb<exit>:
; CHECK-NEXT:    IR   %red.next.lcssa = phi i64 [ %red.next, %loop ] (extra operand: vp<[[VP9]]> from middle.block)
; CHECK-NEXT:  No successors
; CHECK-EMPTY:
; CHECK-NEXT:  ir-bb<scalar.ph>:
; CHECK-NEXT:  Successor(s): ir-bb<loop>
; CHECK-EMPTY:
; CHECK-NEXT:  ir-bb<loop>:
; CHECK-NEXT:    IR   %iv = phi i64 [ 0, %scalar.ph ], [ %iv.next, %loop ] (extra operand: ir<0> from ir-bb<scalar.ph>)
; CHECK-NEXT:    IR   %red = phi i64 [ 5, %scalar.ph ], [ %red.next, %loop ] (extra operand: ir<5> from ir-bb<scalar.ph>)
; CHECK-NEXT:    IR   %gep = getelementptr inbounds i64, ptr %A, i64 %iv
; CHECK-NEXT:    IR   %l = load i64, ptr %gep, align 4
; CHECK-NEXT:    IR   %red.next = add i64 %red, %l
; CHECK-NEXT:    IR   %iv.next = add i64 %iv, 1
; CHECK-NEXT:    IR   %ec = icmp eq i64 %iv.next, %n
; CHECK-NEXT:  No successors
; CHECK-NEXT:  }
;
; CHECK-LABEL: VPlan for loop in 'resume_values'
; CHECK:  VPlan 'Final VPlan for VF={4},UF={1}' {
; CHECK-NEXT:  Live-in ir<%n> = original trip-count
; CHECK-EMPTY:
; CHECK-NEXT:  ir-bb<entry>:
; CHECK-NEXT:    IR   %min.iters.check = icmp ult i64 %n, 4
; CHECK-NEXT:  Successor(s): ir-bb<vec.epilog.scalar.ph>, ir-bb<vector.main.loop.iter.check>
; CHECK-EMPTY:
; CHECK-NEXT:  ir-bb<vector.main.loop.iter.check>:
; CHECK-NEXT:  Successor(s): vec.epilog.ph, ir-bb<vector.ph>
; CHECK-EMPTY:
; CHECK-NEXT:  ir-bb<vector.ph>:
; CHECK-NEXT:  Successor(s): ir-bb<vector.body>
; CHECK-EMPTY:
; CHECK-NEXT:  ir-bb<vector.body>:
; CHECK-NEXT:  Successor(s): ir-bb<middle.block>
; CHECK-EMPTY:
; CHECK-NEXT:  ir-bb<middle.block>:
; CHECK-NEXT:  Successor(s): ir-bb<vec.epilog.iter.check>
; CHECK-EMPTY:
; CHECK-NEXT:  ir-bb<vec.epilog.iter.check>:
; CHECK-NEXT:    EMIT vp<%min.epilog.iters.check> = icmp ult ir<%0>, ir<4>
; CHECK-NEXT:    EMIT branch-on-cond vp<%min.epilog.iters.check>
; CHECK-NEXT:  Successor(s): ir-bb<vec.epilog.scalar.ph>, vec.epilog.ph
; CHECK-EMPTY:
; CHECK-NEXT:  vec.epilog.ph:
; CHECK-NEXT:    EMIT-SCALAR vp<%vec.epilog.resume.val> = phi [ ir<%n.vec>, ir-bb<vec.epilog.iter.check> ], [ ir<0>, ir-bb<vector.main.loop.iter.check> ]
; CHECK-NEXT:    EMIT-SCALAR vp<%bc.merge.rdx> = phi [ ir<%4>, ir-bb<vec.epilog.iter.check> ], [ ir<5>, ir-bb<vector.main.loop.iter.check> ]
; CHECK-NEXT:    EMIT vp<[[VP3:%[0-9]+]]> = and ir<%n>, ir<3>
; CHECK-NEXT:    EMIT vp<%n.vec> = sub ir<%n>, vp<[[VP3]]>
; CHECK-NEXT:    EMIT vp<[[VP4:%[0-9]+]]> = reduction-start-vector vp<%bc.merge.rdx>, ir<0>, ir<1>
; CHECK-NEXT:  Successor(s): vec.epilog.vector.body
; CHECK-EMPTY:
; CHECK-NEXT:  vec.epilog.vector.body:
; CHECK-NEXT:    EMIT-SCALAR vp<%index> = phi [ vp<%vec.epilog.resume.val>, vec.epilog.ph ], [ vp<%index.next>, vec.epilog.vector.body ]
; CHECK-NEXT:    WIDEN-REDUCTION-PHI ir<%red> = phi (add) vp<[[VP4]]>, ir<%red.next>
; CHECK-NEXT:    CLONE ir<%gep> = getelementptr inbounds ir<%A>, vp<%index>
; CHECK-NEXT:    WIDEN ir<%l> = load ir<%gep>
; CHECK-NEXT:    WIDEN ir<%red.next> = add ir<%red>, ir<%l>
; CHECK-NEXT:    EMIT vp<%index.next> = add nuw vp<%index>, ir<4>
; CHECK-NEXT:    EMIT vp<[[VP5:%[0-9]+]]> = icmp eq vp<%index.next>, vp<%n.vec>
; CHECK-NEXT:    EMIT branch-on-cond vp<[[VP5]]>
; CHECK-NEXT:  Successor(s): vec.epilog.middle.block, vec.epilog.vector.body
; CHECK-EMPTY:
; CHECK-NEXT:  vec.epilog.middle.block:
; CHECK-NEXT:    EMIT vp<[[VP7:%[0-9]+]]> = compute-reduction-result (add) ir<%red.next>
; CHECK-NEXT:    EMIT vp<%cmp.n> = icmp eq ir<%n>, vp<%n.vec>
; CHECK-NEXT:    EMIT branch-on-cond vp<%cmp.n>
; CHECK-NEXT:  Successor(s): ir-bb<exit>, ir-bb<vec.epilog.scalar.ph>
; CHECK-EMPTY:
; CHECK-NEXT:  ir-bb<exit>:
; CHECK-NEXT:    IR   %red.next.lcssa = phi i64 [ %red.next, %loop ], [ %4, %middle.block ] (extra operand: vp<[[VP7]]> from vec.epilog.middle.block)
; CHECK-NEXT:  No successors
; CHECK-EMPTY:
; CHECK-NEXT:  ir-bb<vec.epilog.scalar.ph>:
; CHECK-NEXT:    EMIT-SCALAR vp<%bc.resume.val> = phi [ vp<%n.vec>, vec.epilog.middle.block ], [ ir<%n.vec>, ir-bb<vec.epilog.iter.check> ], [ ir<0>, ir-bb<entry> ]
; CHECK-NEXT:    EMIT-SCALAR vp<%bc.merge.rdx>.1 = phi [ vp<[[VP7]]>, vec.epilog.middle.block ], [ ir<%4>, ir-bb<vec.epilog.iter.check> ], [ ir<5>, ir-bb<entry> ]
; CHECK-NEXT:  Successor(s): ir-bb<loop>
; CHECK-EMPTY:
; CHECK-NEXT:  ir-bb<loop>:
; CHECK-NEXT:    IR   %iv = phi i64 [ 0, %vec.epilog.scalar.ph ], [ %iv.next, %loop ] (extra operand: vp<%bc.resume.val> from ir-bb<vec.epilog.scalar.ph>)
; CHECK-NEXT:    IR   %red = phi i64 [ 5, %vec.epilog.scalar.ph ], [ %red.next, %loop ] (extra operand: vp<%bc.merge.rdx>.1 from ir-bb<vec.epilog.scalar.ph>)
; CHECK-NEXT:    IR   %gep = getelementptr inbounds i64, ptr %A, i64 %iv
; CHECK-NEXT:    IR   %l = load i64, ptr %gep, align 4
; CHECK-NEXT:    IR   %red.next = add i64 %red, %l
; CHECK-NEXT:    IR   %iv.next = add i64 %iv, 1
; CHECK-NEXT:    IR   %ec = icmp eq i64 %iv.next, %n
; CHECK-NEXT:  No successors
; CHECK-NEXT:  }
;
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %red = phi i64 [ 5, %entry ], [ %red.next, %loop ]
  %gep = getelementptr inbounds i64, ptr %A, i64 %iv
  %l = load i64, ptr %gep
  %red.next = add i64 %red, %l
  %iv.next = add i64 %iv, 1
  %ec = icmp eq i64 %iv.next, %n
  br i1 %ec, label %exit, label %loop

exit:
  ret i64 %red.next
}

; Same, but with SCEV and memory runtime checks, which also bypass both vector
; loops.
define i64 @bypass_blocks(ptr %A, ptr %B, i32 %n) {
; CHECK-LABEL: VPlan for loop in 'bypass_blocks'
; CHECK:  VPlan 'Final VPlan for VF={8},UF={1}' {
; CHECK-NEXT:  Live-in ir<%n> = original trip-count
; CHECK-EMPTY:
; CHECK-NEXT:  ir-bb<entry>:
; CHECK-NEXT:    IR   %A2 = ptrtoaddr ptr %A to i64
; CHECK-NEXT:    IR   %B1 = ptrtoaddr ptr %B to i64
; CHECK-NEXT:    EMIT vp<%min.iters.check> = icmp ult ir<%n>, ir<4>
; CHECK-NEXT:    EMIT branch-on-cond vp<%min.iters.check>
; CHECK-NEXT:  Successor(s): ir-bb<scalar.ph>, ir-bb<vector.scevcheck>
; CHECK-EMPTY:
; CHECK-NEXT:  ir-bb<vector.scevcheck>:
; CHECK-NEXT:    IR   %0 = add i32 %n, -1
; CHECK-NEXT:    IR   %1 = icmp slt i32 %0, 0
; CHECK-NEXT:    EMIT branch-on-cond ir<%1>
; CHECK-NEXT:  Successor(s): ir-bb<scalar.ph>, ir-bb<vector.memcheck>
; CHECK-EMPTY:
; CHECK-NEXT:  ir-bb<vector.memcheck>:
; CHECK-NEXT:    IR   %2 = sub i64 %B1, %A2
; CHECK-NEXT:    IR   %3 = sub i64 %2, 1
; CHECK-NEXT:    IR   %diff.check = icmp ult i64 %3, 63
; CHECK-NEXT:    EMIT branch-on-cond ir<%diff.check>
; CHECK-NEXT:  Successor(s): ir-bb<scalar.ph>, vector.main.loop.iter.check
; CHECK-EMPTY:
; CHECK-NEXT:  vector.main.loop.iter.check:
; CHECK-NEXT:    EMIT vp<%min.iters.check>.1 = icmp ult ir<%n>, ir<8>
; CHECK-NEXT:    EMIT branch-on-cond vp<%min.iters.check>.1
; CHECK-NEXT:  Successor(s): ir-bb<scalar.ph>, vector.ph
; CHECK-EMPTY:
; CHECK-NEXT:  vector.ph:
; CHECK-NEXT:    EMIT vp<[[VP6:%[0-9]+]]> = and ir<%n>, ir<7>
; CHECK-NEXT:    EMIT vp<%n.vec> = sub ir<%n>, vp<[[VP6]]>
; CHECK-NEXT:    EMIT vp<[[VP7:%[0-9]+]]> = reduction-start-vector ir<0>, ir<0>, ir<1>
; CHECK-NEXT:  Successor(s): vector.body
; CHECK-EMPTY:
; CHECK-NEXT:  vector.body:
; CHECK-NEXT:    EMIT-SCALAR vp<%index> = phi [ ir<0>, vector.ph ], [ vp<%index.next>, vector.body ]
; CHECK-NEXT:    WIDEN-REDUCTION-PHI ir<%red> = phi (add) vp<[[VP7]]>, ir<%red.next>
; CHECK-NEXT:    EMIT-SCALAR ir<%iv.ext> = sext vp<%index> to i64
; CHECK-NEXT:    CLONE ir<%gep.a> = getelementptr inbounds ir<%A>, ir<%iv.ext>
; CHECK-NEXT:    WIDEN ir<%l> = load ir<%gep.a>
; CHECK-NEXT:    WIDEN ir<%red.next> = add ir<%red>, ir<%l>
; CHECK-NEXT:    CLONE ir<%gep.b> = getelementptr inbounds ir<%B>, ir<%iv.ext>
; CHECK-NEXT:    WIDEN store ir<%gep.b>, ir<%l>
; CHECK-NEXT:    EMIT vp<%index.next> = add nuw vp<%index>, ir<8>
; CHECK-NEXT:    EMIT vp<[[VP8:%[0-9]+]]> = icmp eq vp<%index.next>, vp<%n.vec>
; CHECK-NEXT:    EMIT branch-on-cond vp<[[VP8]]>
; CHECK-NEXT:  Successor(s): middle.block, vector.body
; CHECK-EMPTY:
; CHECK-NEXT:  middle.block:
; CHECK-NEXT:    EMIT-SCALAR vp<%{{.+}}> = resume-for-epilogue vp<%n.vec>, ir<0>
; CHECK-NEXT:    EMIT vp<[[VP11:%[0-9]+]]> = compute-reduction-result (add) ir<%red.next>
; CHECK-NEXT:    EMIT vp<%cmp.n> = icmp eq ir<%n>, vp<%n.vec>
; CHECK-NEXT:    EMIT-SCALAR vp<%{{.+}}> = resume-for-epilogue vp<%n.vec>, ir<0>
; CHECK-NEXT:    EMIT-SCALAR vp<%{{.+}}> = resume-for-epilogue vp<[[VP11]]>, ir<0>
; CHECK-NEXT:    EMIT branch-on-cond vp<%cmp.n>
; CHECK-NEXT:  Successor(s): ir-bb<exit>, ir-bb<scalar.ph>
; CHECK-EMPTY:
; CHECK-NEXT:  ir-bb<exit>:
; CHECK-NEXT:    IR   %red.next.lcssa = phi i64 [ %red.next, %loop ] (extra operand: vp<[[VP11]]> from middle.block)
; CHECK-NEXT:  No successors
; CHECK-EMPTY:
; CHECK-NEXT:  ir-bb<scalar.ph>:
; CHECK-NEXT:  Successor(s): ir-bb<loop>
; CHECK-EMPTY:
; CHECK-NEXT:  ir-bb<loop>:
; CHECK-NEXT:    IR   %iv = phi i32 [ 0, %scalar.ph ], [ %iv.next, %loop ] (extra operand: ir<0> from ir-bb<scalar.ph>)
; CHECK-NEXT:    IR   %red = phi i64 [ 0, %scalar.ph ], [ %red.next, %loop ] (extra operand: ir<0> from ir-bb<scalar.ph>)
; CHECK-NEXT:    IR   %iv.ext = sext i32 %iv to i64
; CHECK-NEXT:    IR   %gep.a = getelementptr inbounds i64, ptr %A, i64 %iv.ext
; CHECK-NEXT:    IR   %l = load i64, ptr %gep.a, align 4
; CHECK-NEXT:    IR   %red.next = add i64 %red, %l
; CHECK-NEXT:    IR   %gep.b = getelementptr inbounds i64, ptr %B, i64 %iv.ext
; CHECK-NEXT:    IR   store i64 %l, ptr %gep.b, align 4
; CHECK-NEXT:    IR   %iv.next = add i32 %iv, 1
; CHECK-NEXT:    IR   %ec = icmp eq i32 %iv.next, %n
; CHECK-NEXT:  No successors
; CHECK-NEXT:  }
;
; CHECK-LABEL: VPlan for loop in 'bypass_blocks'
; CHECK:  VPlan 'Final VPlan for VF={4},UF={1}' {
; CHECK-NEXT:  Live-in ir<%n> = original trip-count
; CHECK-EMPTY:
; CHECK-NEXT:  ir-bb<entry>:
; CHECK-NEXT:    IR   %A2 = ptrtoaddr ptr %A to i64
; CHECK-NEXT:    IR   %B1 = ptrtoaddr ptr %B to i64
; CHECK-NEXT:    IR   %min.iters.check = icmp ult i32 %n, 4
; CHECK-NEXT:  Successor(s): ir-bb<vec.epilog.scalar.ph>, ir-bb<vector.scevcheck>
; CHECK-EMPTY:
; CHECK-NEXT:  ir-bb<vector.scevcheck>:
; CHECK-NEXT:  Successor(s): ir-bb<vec.epilog.scalar.ph>, ir-bb<vector.memcheck>
; CHECK-EMPTY:
; CHECK-NEXT:  ir-bb<vector.memcheck>:
; CHECK-NEXT:  Successor(s): ir-bb<vec.epilog.scalar.ph>, ir-bb<vector.main.loop.iter.check>
; CHECK-EMPTY:
; CHECK-NEXT:  ir-bb<vector.main.loop.iter.check>:
; CHECK-NEXT:  Successor(s): vec.epilog.ph, ir-bb<vector.ph>
; CHECK-EMPTY:
; CHECK-NEXT:  ir-bb<vector.ph>:
; CHECK-NEXT:  Successor(s): ir-bb<vector.body>
; CHECK-EMPTY:
; CHECK-NEXT:  ir-bb<vector.body>:
; CHECK-NEXT:  Successor(s): ir-bb<middle.block>
; CHECK-EMPTY:
; CHECK-NEXT:  ir-bb<middle.block>:
; CHECK-NEXT:  Successor(s): ir-bb<vec.epilog.iter.check>
; CHECK-EMPTY:
; CHECK-NEXT:  ir-bb<vec.epilog.iter.check>:
; CHECK-NEXT:    EMIT vp<%min.epilog.iters.check> = icmp ult ir<%4>, ir<4>
; CHECK-NEXT:    EMIT branch-on-cond vp<%min.epilog.iters.check>
; CHECK-NEXT:  Successor(s): ir-bb<vec.epilog.scalar.ph>, vec.epilog.ph
; CHECK-EMPTY:
; CHECK-NEXT:  vec.epilog.ph:
; CHECK-NEXT:    EMIT-SCALAR vp<%vec.epilog.resume.val> = phi [ ir<%n.vec>, ir-bb<vec.epilog.iter.check> ], [ ir<0>, ir-bb<vector.main.loop.iter.check> ]
; CHECK-NEXT:    EMIT-SCALAR vp<%bc.merge.rdx> = phi [ ir<%10>, ir-bb<vec.epilog.iter.check> ], [ ir<0>, ir-bb<vector.main.loop.iter.check> ]
; CHECK-NEXT:    EMIT vp<[[VP3:%[0-9]+]]> = and ir<%n>, ir<3>
; CHECK-NEXT:    EMIT vp<%n.vec> = sub ir<%n>, vp<[[VP3]]>
; CHECK-NEXT:    EMIT vp<[[VP4:%[0-9]+]]> = reduction-start-vector vp<%bc.merge.rdx>, ir<0>, ir<1>
; CHECK-NEXT:  Successor(s): vec.epilog.vector.body
; CHECK-EMPTY:
; CHECK-NEXT:  vec.epilog.vector.body:
; CHECK-NEXT:    EMIT-SCALAR vp<%index> = phi [ vp<%vec.epilog.resume.val>, vec.epilog.ph ], [ vp<%index.next>, vec.epilog.vector.body ]
; CHECK-NEXT:    WIDEN-REDUCTION-PHI ir<%red> = phi (add) vp<[[VP4]]>, ir<%red.next>
; CHECK-NEXT:    EMIT-SCALAR ir<%iv.ext> = sext vp<%index> to i64
; CHECK-NEXT:    CLONE ir<%gep.a> = getelementptr inbounds ir<%A>, ir<%iv.ext>
; CHECK-NEXT:    WIDEN ir<%l> = load ir<%gep.a>
; CHECK-NEXT:    WIDEN ir<%red.next> = add ir<%red>, ir<%l>
; CHECK-NEXT:    CLONE ir<%gep.b> = getelementptr inbounds ir<%B>, ir<%iv.ext>
; CHECK-NEXT:    WIDEN store ir<%gep.b>, ir<%l>
; CHECK-NEXT:    EMIT vp<%index.next> = add nuw vp<%index>, ir<4>
; CHECK-NEXT:    EMIT vp<[[VP5:%[0-9]+]]> = icmp eq vp<%index.next>, vp<%n.vec>
; CHECK-NEXT:    EMIT branch-on-cond vp<[[VP5]]>
; CHECK-NEXT:  Successor(s): vec.epilog.middle.block, vec.epilog.vector.body
; CHECK-EMPTY:
; CHECK-NEXT:  vec.epilog.middle.block:
; CHECK-NEXT:    EMIT vp<[[VP7:%[0-9]+]]> = compute-reduction-result (add) ir<%red.next>
; CHECK-NEXT:    EMIT vp<%cmp.n> = icmp eq ir<%n>, vp<%n.vec>
; CHECK-NEXT:    EMIT branch-on-cond vp<%cmp.n>
; CHECK-NEXT:  Successor(s): ir-bb<exit>, ir-bb<vec.epilog.scalar.ph>
; CHECK-EMPTY:
; CHECK-NEXT:  ir-bb<exit>:
; CHECK-NEXT:    IR   %red.next.lcssa = phi i64 [ %red.next, %loop ], [ %10, %middle.block ] (extra operand: vp<[[VP7]]> from vec.epilog.middle.block)
; CHECK-NEXT:  No successors
; CHECK-EMPTY:
; CHECK-NEXT:  ir-bb<vec.epilog.scalar.ph>:
; CHECK-NEXT:    EMIT-SCALAR vp<%bc.resume.val> = phi [ vp<%n.vec>, vec.epilog.middle.block ], [ ir<%n.vec>, ir-bb<vec.epilog.iter.check> ], [ ir<0>, ir-bb<vector.memcheck> ], [ ir<0>, ir-bb<vector.scevcheck> ], [ ir<0>, ir-bb<entry> ]
; CHECK-NEXT:    EMIT-SCALAR vp<%bc.merge.rdx>.1 = phi [ vp<[[VP7]]>, vec.epilog.middle.block ], [ ir<%10>, ir-bb<vec.epilog.iter.check> ], [ ir<0>, ir-bb<vector.memcheck> ], [ ir<0>, ir-bb<vector.scevcheck> ], [ ir<0>, ir-bb<entry> ]
; CHECK-NEXT:  Successor(s): ir-bb<loop>
; CHECK-EMPTY:
; CHECK-NEXT:  ir-bb<loop>:
; CHECK-NEXT:    IR   %iv = phi i32 [ 0, %vec.epilog.scalar.ph ], [ %iv.next, %loop ] (extra operand: vp<%bc.resume.val> from ir-bb<vec.epilog.scalar.ph>)
; CHECK-NEXT:    IR   %red = phi i64 [ 0, %vec.epilog.scalar.ph ], [ %red.next, %loop ] (extra operand: vp<%bc.merge.rdx>.1 from ir-bb<vec.epilog.scalar.ph>)
; CHECK-NEXT:    IR   %iv.ext = sext i32 %iv to i64
; CHECK-NEXT:    IR   %gep.a = getelementptr inbounds i64, ptr %A, i64 %iv.ext
; CHECK-NEXT:    IR   %l = load i64, ptr %gep.a, align 4
; CHECK-NEXT:    IR   %red.next = add i64 %red, %l
; CHECK-NEXT:    IR   %gep.b = getelementptr inbounds i64, ptr %B, i64 %iv.ext
; CHECK-NEXT:    IR   store i64 %l, ptr %gep.b, align 4
; CHECK-NEXT:    IR   %iv.next = add i32 %iv, 1
; CHECK-NEXT:    IR   %ec = icmp eq i32 %iv.next, %n
; CHECK-NEXT:  No successors
; CHECK-NEXT:  }
;
entry:
  br label %loop

loop:
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %loop ]
  %red = phi i64 [ 0, %entry ], [ %red.next, %loop ]
  %iv.ext = sext i32 %iv to i64
  %gep.a = getelementptr inbounds i64, ptr %A, i64 %iv.ext
  %l = load i64, ptr %gep.a
  %red.next = add i64 %red, %l
  %gep.b = getelementptr inbounds i64, ptr %B, i64 %iv.ext
  store i64 %l, ptr %gep.b
  %iv.next = add i32 %iv, 1
  %ec = icmp eq i32 %iv.next, %n
  br i1 %ec, label %exit, label %loop

exit:
  ret i64 %red.next
}

; The main vector loop covers all 16 iterations, so the checks bypassing it are
; constant and fold: the middle block branches to the exit block only and the
; scalar preheader has no resume phis. The memory runtime check keeps the scalar
; loop reachable.
define void @all_iterations_in_main_loop_with_memcheck(ptr %dst, ptr %src) {
; CHECK-LABEL: VPlan for loop in 'all_iterations_in_main_loop_with_memcheck'
; CHECK:  VPlan 'Final VPlan for VF={8},UF={1}' {
; CHECK-NEXT:  Live-in ir<16> = vector-trip-count
; CHECK-EMPTY:
; CHECK-NEXT:  ir-bb<entry>:
; CHECK-NEXT:    IR   %src2 = ptrtoaddr ptr %src to i64
; CHECK-NEXT:    IR   %dst1 = ptrtoaddr ptr %dst to i64
; CHECK-NEXT:  Successor(s): ir-bb<vector.memcheck>
; CHECK-EMPTY:
; CHECK-NEXT:  ir-bb<vector.memcheck>:
; CHECK-NEXT:    IR   %0 = sub i64 %dst1, %src2
; CHECK-NEXT:    IR   %1 = sub i64 %0, 1
; CHECK-NEXT:    IR   %diff.check = icmp ult i64 %1, 31
; CHECK-NEXT:    EMIT branch-on-cond ir<%diff.check>
; CHECK-NEXT:  Successor(s): ir-bb<scalar.ph>, vector.main.loop.iter.check
; CHECK-EMPTY:
; CHECK-NEXT:  vector.main.loop.iter.check:
; CHECK-NEXT:  Successor(s): vector.ph
; CHECK-EMPTY:
; CHECK-NEXT:  vector.ph:
; CHECK-NEXT:  Successor(s): vector.body
; CHECK-EMPTY:
; CHECK-NEXT:  vector.body:
; CHECK-NEXT:    EMIT-SCALAR vp<%index> = phi [ ir<0>, vector.ph ], [ vp<%index.next>, vector.body ]
; CHECK-NEXT:    CLONE ir<%gep.src> = getelementptr inbounds ir<%src>, vp<%index>
; CHECK-NEXT:    WIDEN ir<%l> = load ir<%gep.src>
; CHECK-NEXT:    WIDEN ir<%add> = add ir<%l>, ir<1>
; CHECK-NEXT:    CLONE ir<%gep.dst> = getelementptr inbounds ir<%dst>, vp<%index>
; CHECK-NEXT:    WIDEN store ir<%gep.dst>, ir<%add>
; CHECK-NEXT:    EMIT vp<%index.next> = add nuw vp<%index>, ir<8>
; CHECK-NEXT:    EMIT vp<[[VP2:%[0-9]+]]> = icmp eq vp<%index.next>, ir<16>
; CHECK-NEXT:    EMIT branch-on-cond vp<[[VP2]]>
; CHECK-NEXT:  Successor(s): middle.block, vector.body
; CHECK-EMPTY:
; CHECK-NEXT:  middle.block:
; CHECK-NEXT:    EMIT-SCALAR vp<%{{.+}}> = resume-for-epilogue ir<16>, ir<0>
; CHECK-NEXT:    EMIT-SCALAR vp<%{{.+}}> = resume-for-epilogue ir<16>, ir<0>
; CHECK-NEXT:  Successor(s): ir-bb<exit>
; CHECK-EMPTY:
; CHECK-NEXT:  ir-bb<exit>:
; CHECK-NEXT:  No successors
; CHECK-EMPTY:
; CHECK-NEXT:  ir-bb<scalar.ph>:
; CHECK-NEXT:  Successor(s): ir-bb<loop>
; CHECK-EMPTY:
; CHECK-NEXT:  ir-bb<loop>:
; CHECK-NEXT:    IR   %iv = phi i64 [ 0, %scalar.ph ], [ %iv.next, %loop ] (extra operand: ir<0> from ir-bb<scalar.ph>)
; CHECK-NEXT:    IR   %gep.src = getelementptr inbounds i32, ptr %src, i64 %iv
; CHECK-NEXT:    IR   %l = load i32, ptr %gep.src, align 4
; CHECK-NEXT:    IR   %add = add i32 %l, 1
; CHECK-NEXT:    IR   %gep.dst = getelementptr inbounds i32, ptr %dst, i64 %iv
; CHECK-NEXT:    IR   store i32 %add, ptr %gep.dst, align 4
; CHECK-NEXT:    IR   %iv.next = add i64 %iv, 1
; CHECK-NEXT:    IR   %ec = icmp eq i64 %iv.next, 16
; CHECK-NEXT:  No successors
; CHECK-NEXT:  }
; CHECK-NOT:   VPlan for loop in 'all_iterations_in_main_loop_with_memcheck'
;
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %gep.src = getelementptr inbounds i32, ptr %src, i64 %iv
  %l = load i32, ptr %gep.src, align 4
  %add = add i32 %l, 1
  %gep.dst = getelementptr inbounds i32, ptr %dst, i64 %iv
  store i32 %add, ptr %gep.dst, align 4
  %iv.next = add i64 %iv, 1
  %ec = icmp eq i64 %iv.next, 16
  br i1 %ec, label %exit, label %loop

exit:
  ret void
}

; Same as @all_iterations_in_main_loop_with_memcheck, but the SCEV runtime check
; keeps the scalar loop reachable.
define void @all_iterations_in_main_loop_with_scevcheck(ptr %p, i32 %off) {
; CHECK-LABEL: VPlan for loop in 'all_iterations_in_main_loop_with_scevcheck'
; CHECK:  VPlan 'Final VPlan for VF={8},UF={1}' {
; CHECK-NEXT:  Live-in ir<16> = vector-trip-count
; CHECK-EMPTY:
; CHECK-NEXT:  ir-bb<entry>:
; CHECK-NEXT:  Successor(s): ir-bb<vector.scevcheck>
; CHECK-EMPTY:
; CHECK-NEXT:  ir-bb<vector.scevcheck>:
; CHECK-NEXT:    IR   %0 = add i32 %off, 15
; CHECK-NEXT:    IR   %1 = icmp ult i32 %0, %off
; CHECK-NEXT:    EMIT branch-on-cond ir<%1>
; CHECK-NEXT:  Successor(s): ir-bb<scalar.ph>, vector.main.loop.iter.check
; CHECK-EMPTY:
; CHECK-NEXT:  vector.main.loop.iter.check:
; CHECK-NEXT:  Successor(s): vector.ph
; CHECK-EMPTY:
; CHECK-NEXT:  vector.ph:
; CHECK-NEXT:    EMIT vp<[[VP2:%[0-9]+]]> = add ir<%off>, ir<16>
; CHECK-NEXT:  Successor(s): vector.body
; CHECK-EMPTY:
; CHECK-NEXT:  vector.body:
; CHECK-NEXT:    EMIT-SCALAR vp<%index> = phi [ ir<0>, vector.ph ], [ vp<%index.next>, vector.body ]
; CHECK-NEXT:    EMIT-SCALAR vp<[[VP3:%[0-9]+]]> = trunc vp<%index> to i32
; CHECK-NEXT:    EMIT vp<[[VP4:%[0-9]+]]> = add ir<%off>, vp<[[VP3]]>
; CHECK-NEXT:    EMIT-SCALAR ir<%idx> = zext vp<[[VP4]]> to i64
; CHECK-NEXT:    CLONE ir<%gep> = getelementptr inbounds ir<%p>, ir<%idx>
; CHECK-NEXT:    WIDEN store ir<%gep>, ir<1>
; CHECK-NEXT:    EMIT vp<%index.next> = add nuw vp<%index>, ir<8>
; CHECK-NEXT:    EMIT vp<[[VP5:%[0-9]+]]> = icmp eq vp<%index.next>, ir<16>
; CHECK-NEXT:    EMIT branch-on-cond vp<[[VP5]]>
; CHECK-NEXT:  Successor(s): middle.block, vector.body
; CHECK-EMPTY:
; CHECK-NEXT:  middle.block:
; CHECK-NEXT:    EMIT-SCALAR vp<%{{.+}}> = resume-for-epilogue ir<16>, ir<0>
; CHECK-NEXT:    EMIT-SCALAR vp<%{{.+}}> = resume-for-epilogue ir<16>, ir<0>
; CHECK-NEXT:    EMIT-SCALAR vp<%{{.+}}> = resume-for-epilogue vp<[[VP2]]>, ir<%off>
; CHECK-NEXT:  Successor(s): ir-bb<exit>
; CHECK-EMPTY:
; CHECK-NEXT:  ir-bb<exit>:
; CHECK-NEXT:  No successors
; CHECK-EMPTY:
; CHECK-NEXT:  ir-bb<scalar.ph>:
; CHECK-NEXT:  Successor(s): ir-bb<loop>
; CHECK-EMPTY:
; CHECK-NEXT:  ir-bb<loop>:
; CHECK-NEXT:    IR   %iv = phi i64 [ 0, %scalar.ph ], [ %iv.next, %loop ] (extra operand: ir<0> from ir-bb<scalar.ph>)
; CHECK-NEXT:    IR   %iv.narrow = phi i32 [ %off, %scalar.ph ], [ %iv.narrow.next, %loop ] (extra operand: ir<%off> from ir-bb<scalar.ph>)
; CHECK-NEXT:    IR   %idx = zext i32 %iv.narrow to i64
; CHECK-NEXT:    IR   %gep = getelementptr inbounds i32, ptr %p, i64 %idx
; CHECK-NEXT:    IR   store i32 1, ptr %gep, align 4
; CHECK-NEXT:    IR   %iv.narrow.next = add i32 %iv.narrow, 1
; CHECK-NEXT:    IR   %iv.next = add i64 %iv, 1
; CHECK-NEXT:    IR   %ec = icmp eq i64 %iv.next, 16
; CHECK-NEXT:  No successors
; CHECK-NEXT:  }
; CHECK-NOT:   VPlan for loop in 'all_iterations_in_main_loop_with_scevcheck'
;
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %iv.narrow = phi i32 [ %off, %entry ], [ %iv.narrow.next, %loop ]
  %idx = zext i32 %iv.narrow to i64
  %gep = getelementptr inbounds i32, ptr %p, i64 %idx
  store i32 1, ptr %gep, align 4
  %iv.narrow.next = add i32 %iv.narrow, 1
  %iv.next = add i64 %iv, 1
  %ec = icmp eq i64 %iv.next, 16
  br i1 %ec, label %exit, label %loop

exit:
  ret void
}

; The iteration count check for the main vector loop always bypasses it, so the
; main vector loop is dead
define void @dead_main_vector_loop(ptr %dst, i64 %n) {
; CHECK-LABEL: VPlan for loop in 'dead_main_vector_loop'
; CHECK:  VPlan 'Final VPlan for VF={8},UF={1}' {
; CHECK-NEXT:  Live-in ir<%clamped> = original trip-count
; CHECK-EMPTY:
; CHECK-NEXT:  ir-bb<entry>:
; CHECK-NEXT:    IR   %clamped = call i64 @llvm.umin.i64(i64 %n, i64 4)
; CHECK-NEXT:    EMIT vp<%min.iters.check> = icmp ult ir<%clamped>, ir<4>
; CHECK-NEXT:    EMIT branch-on-cond vp<%min.iters.check>
; CHECK-NEXT:  Successor(s): ir-bb<scalar.ph>, vector.main.loop.iter.check
; CHECK-EMPTY:
; CHECK-NEXT:  vector.main.loop.iter.check:
; CHECK-NEXT:  Successor(s): ir-bb<scalar.ph>
; CHECK-EMPTY:
; CHECK-NEXT:  ir-bb<scalar.ph>:
; CHECK-NEXT:  Successor(s): ir-bb<loop>
; CHECK-EMPTY:
; CHECK-NEXT:  ir-bb<loop>:
; CHECK-NEXT:    IR   %iv = phi i64 [ 0, %scalar.ph ], [ %iv.next, %loop ] (extra operand: ir<0> from ir-bb<scalar.ph>)
; CHECK-NEXT:    IR   %gep = getelementptr inbounds i32, ptr %dst, i64 %iv
; CHECK-NEXT:    IR   store i32 1, ptr %gep, align 4
; CHECK-NEXT:    IR   %iv.next = add i64 %iv, 1
; CHECK-NEXT:    IR   %ec = icmp eq i64 %iv.next, %clamped
; CHECK-NEXT:  No successors
; CHECK-NEXT:  }
; CHECK-NOT:   VPlan for loop in 'dead_main_vector_loop'
;
entry:
  %clamped = call i64 @llvm.umin.i64(i64 %n, i64 4)
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %gep = getelementptr inbounds i32, ptr %dst, i64 %iv
  store i32 1, ptr %gep, align 4
  %iv.next = add i64 %iv, 1
  %ec = icmp eq i64 %iv.next, %clamped
  br i1 %ec, label %exit, label %loop

exit:
  ret void
}

; The vectorized loop is nested, so the block the epilogue plan is entered from
; is the header of the outer loop.
define i32 @nested_loop(ptr noalias %p, ptr noalias %end, ptr noalias %dst, i64 %m) {
; CHECK-LABEL: VPlan for loop in 'nested_loop'
; CHECK:  VPlan 'Final VPlan for VF={8},UF={1}' {
; CHECK-NEXT:  Live-in ir<%3> = original trip-count
; CHECK-EMPTY:
; CHECK-NEXT:  ir-bb<outer.header>:
; CHECK-NEXT:    IR   %j = phi i64 [ 0, %entry ], [ %j.next, %outer.latch ]
; CHECK-NEXT:    EMIT vp<%min.iters.check> = icmp ult ir<%3>, ir<4>
; CHECK-NEXT:    EMIT branch-on-cond vp<%min.iters.check>
; CHECK-NEXT:  Successor(s): ir-bb<scalar.ph>, vector.main.loop.iter.check
; CHECK-EMPTY:
; CHECK-NEXT:  vector.main.loop.iter.check:
; CHECK-NEXT:    EMIT vp<%min.iters.check>.1 = icmp ult ir<%3>, ir<8>
; CHECK-NEXT:    EMIT branch-on-cond vp<%min.iters.check>.1
; CHECK-NEXT:  Successor(s): ir-bb<scalar.ph>, vector.ph
; CHECK-EMPTY:
; CHECK-NEXT:  vector.ph:
; CHECK-NEXT:    EMIT vp<[[VP4:%[0-9]+]]> = and ir<%3>, ir<7>
; CHECK-NEXT:    EMIT vp<%n.vec> = sub ir<%3>, vp<[[VP4]]>
; CHECK-NEXT:    EMIT vp<[[VP5:%[0-9]+]]> = mul vp<%n.vec>, ir<24>
; CHECK-NEXT:    EMIT vp<[[VP6:%[0-9]+]]> = ptradd ir<%p>, vp<[[VP5]]>
; CHECK-NEXT:  Successor(s): vector.body
; CHECK-EMPTY:
; CHECK-NEXT:  vector.body:
; CHECK-NEXT:    EMIT-SCALAR vp<%index> = phi [ ir<0>, vector.ph ], [ vp<%index.next>, vector.body ]
; CHECK-NEXT:    EMIT vp<[[VP7:%[0-9]+]]> = mul vp<%index>, ir<24>
; CHECK-NEXT:    vp<[[VP8:%[0-9]+]]> = SCALAR-STEPS vp<[[VP7]]>, ir<24>, ir<8>, ir<7>
; CHECK-NEXT:    EMIT vp<%next.gep> = ptradd ir<%p>, vp<[[VP8]]>
; CHECK-NEXT:    CLONE ir<%l> = load vp<%next.gep>
; CHECK-NEXT:    CLONE store ir<%l>, ir<%dst>
; CHECK-NEXT:    EMIT vp<%index.next> = add nuw vp<%index>, ir<8>
; CHECK-NEXT:    EMIT vp<[[VP9:%[0-9]+]]> = icmp eq vp<%index.next>, vp<%n.vec>
; CHECK-NEXT:    EMIT branch-on-cond vp<[[VP9]]>
; CHECK-NEXT:  Successor(s): middle.block, vector.body
; CHECK-EMPTY:
; CHECK-NEXT:  middle.block:
; CHECK-NEXT:    EMIT-SCALAR vp<%{{.+}}> = resume-for-epilogue vp<%n.vec>, ir<0>
; CHECK-NEXT:    EMIT vp<%cmp.n> = icmp eq ir<%3>, vp<%n.vec>
; CHECK-NEXT:    EMIT-SCALAR vp<%{{.+}}> = resume-for-epilogue vp<[[VP6]]>, ir<%p>
; CHECK-NEXT:    EMIT branch-on-cond vp<%cmp.n>
; CHECK-NEXT:  Successor(s): ir-bb<inner.exit>, ir-bb<scalar.ph>
; CHECK-EMPTY:
; CHECK-NEXT:  ir-bb<inner.exit>:
; CHECK-NEXT:    IR   %cc = icmp ult i64 %j, 3
; CHECK-NEXT:  No successors
; CHECK-EMPTY:
; CHECK-NEXT:  ir-bb<scalar.ph>:
; CHECK-NEXT:  Successor(s): ir-bb<loop>
; CHECK-EMPTY:
; CHECK-NEXT:  ir-bb<loop>:
; CHECK-NEXT:    IR   %q = phi ptr [ %p, %scalar.ph ], [ %q.next, %loop ] (extra operand: ir<%p> from ir-bb<scalar.ph>)
; CHECK-NEXT:    IR   %l = load i32, ptr %q, align 8
; CHECK-NEXT:    IR   store i32 %l, ptr %dst, align 4
; CHECK-NEXT:    IR   %q.next = getelementptr inbounds nuw i8, ptr %q, i64 24
; CHECK-NEXT:    IR   %ec = icmp eq ptr %q.next, %end
; CHECK-NEXT:  No successors
; CHECK-NEXT:  }
;
; CHECK-LABEL: VPlan for loop in 'nested_loop'
; CHECK:  VPlan 'Final VPlan for VF={4},UF={1}' {
; CHECK-NEXT:  Live-in ir<%3> = original trip-count
; CHECK-EMPTY:
; CHECK-NEXT:  ir-bb<outer.header>:
; CHECK-NEXT:    IR   %j = phi i64 [ 0, %entry ], [ %j.next, %outer.latch ]
; CHECK-NEXT:    IR   %min.iters.check = icmp ult i64 %3, 4
; CHECK-NEXT:  Successor(s): ir-bb<vec.epilog.scalar.ph>, ir-bb<vector.main.loop.iter.check>
; CHECK-EMPTY:
; CHECK-NEXT:  ir-bb<vector.main.loop.iter.check>:
; CHECK-NEXT:  Successor(s): vec.epilog.ph, ir-bb<vector.ph>
; CHECK-EMPTY:
; CHECK-NEXT:  ir-bb<vector.ph>:
; CHECK-NEXT:  Successor(s): ir-bb<vector.body>
; CHECK-EMPTY:
; CHECK-NEXT:  ir-bb<vector.body>:
; CHECK-NEXT:  Successor(s): ir-bb<middle.block>
; CHECK-EMPTY:
; CHECK-NEXT:  ir-bb<middle.block>:
; CHECK-NEXT:  Successor(s): ir-bb<vec.epilog.iter.check>
; CHECK-EMPTY:
; CHECK-NEXT:  ir-bb<vec.epilog.iter.check>:
; CHECK-NEXT:    EMIT vp<%min.epilog.iters.check> = icmp ult ir<%4>, ir<4>
; CHECK-NEXT:    EMIT branch-on-cond vp<%min.epilog.iters.check>
; CHECK-NEXT:  Successor(s): ir-bb<vec.epilog.scalar.ph>, vec.epilog.ph
; CHECK-EMPTY:
; CHECK-NEXT:  vec.epilog.ph:
; CHECK-NEXT:    EMIT-SCALAR vp<%vec.epilog.resume.val> = phi [ ir<%n.vec>, ir-bb<vec.epilog.iter.check> ], [ ir<0>, ir-bb<vector.main.loop.iter.check> ]
; CHECK-NEXT:    EMIT vp<[[VP3:%[0-9]+]]> = and ir<%3>, ir<3>
; CHECK-NEXT:    EMIT vp<%n.vec> = sub ir<%3>, vp<[[VP3]]>
; CHECK-NEXT:    EMIT vp<[[VP4:%[0-9]+]]> = mul vp<%n.vec>, ir<24>
; CHECK-NEXT:    EMIT vp<[[VP5:%[0-9]+]]> = ptradd ir<%p>, vp<[[VP4]]>
; CHECK-NEXT:  Successor(s): vec.epilog.vector.body
; CHECK-EMPTY:
; CHECK-NEXT:  vec.epilog.vector.body:
; CHECK-NEXT:    EMIT-SCALAR vp<%index> = phi [ vp<%vec.epilog.resume.val>, vec.epilog.ph ], [ vp<%index.next>, vec.epilog.vector.body ]
; CHECK-NEXT:    EMIT vp<[[VP6:%[0-9]+]]> = mul vp<%index>, ir<24>
; CHECK-NEXT:    vp<[[VP7:%[0-9]+]]> = SCALAR-STEPS vp<[[VP6]]>, ir<24>, ir<4>, ir<3>
; CHECK-NEXT:    EMIT vp<%next.gep> = ptradd ir<%p>, vp<[[VP7]]>
; CHECK-NEXT:    CLONE ir<%l> = load vp<%next.gep>
; CHECK-NEXT:    CLONE store ir<%l>, ir<%dst>
; CHECK-NEXT:    EMIT vp<%index.next> = add nuw vp<%index>, ir<4>
; CHECK-NEXT:    EMIT vp<[[VP8:%[0-9]+]]> = icmp eq vp<%index.next>, vp<%n.vec>
; CHECK-NEXT:    EMIT branch-on-cond vp<[[VP8]]>
; CHECK-NEXT:  Successor(s): vec.epilog.middle.block, vec.epilog.vector.body
; CHECK-EMPTY:
; CHECK-NEXT:  vec.epilog.middle.block:
; CHECK-NEXT:    EMIT vp<%cmp.n> = icmp eq ir<%3>, vp<%n.vec>
; CHECK-NEXT:    EMIT branch-on-cond vp<%cmp.n>
; CHECK-NEXT:  Successor(s): ir-bb<inner.exit>, ir-bb<vec.epilog.scalar.ph>
; CHECK-EMPTY:
; CHECK-NEXT:  ir-bb<inner.exit>:
; CHECK-NEXT:    IR   %cc = icmp ult i64 %j, 3
; CHECK-NEXT:  No successors
; CHECK-EMPTY:
; CHECK-NEXT:  ir-bb<vec.epilog.scalar.ph>:
; CHECK-NEXT:    EMIT-SCALAR vp<%bc.resume.val> = phi [ vp<[[VP5]]>, vec.epilog.middle.block ], [ ir<%6>, ir-bb<vec.epilog.iter.check> ], [ ir<%p>, ir-bb<outer.header> ]
; CHECK-NEXT:  Successor(s): ir-bb<loop>
; CHECK-EMPTY:
; CHECK-NEXT:  ir-bb<loop>:
; CHECK-NEXT:    IR   %q = phi ptr [ %p, %vec.epilog.scalar.ph ], [ %q.next, %loop ] (extra operand: vp<%bc.resume.val> from ir-bb<vec.epilog.scalar.ph>)
; CHECK-NEXT:    IR   %l = load i32, ptr %q, align 8
; CHECK-NEXT:    IR   store i32 %l, ptr %dst, align 4
; CHECK-NEXT:    IR   %q.next = getelementptr inbounds nuw i8, ptr %q, i64 24
; CHECK-NEXT:    IR   %ec = icmp eq ptr %q.next, %end
; CHECK-NEXT:  No successors
; CHECK-NEXT:  }
;
entry:
  br label %outer.header

outer.header:
  %j = phi i64 [ 0, %entry ], [ %j.next, %outer.latch ]
  br label %loop

loop:
  %q = phi ptr [ %p, %outer.header ], [ %q.next, %loop ]
  %l = load i32, ptr %q, align 8
  store i32 %l, ptr %dst, align 4
  %q.next = getelementptr inbounds nuw i8, ptr %q, i64 24
  %ec = icmp eq ptr %q.next, %end
  br i1 %ec, label %inner.exit, label %loop

inner.exit:
  %cc = icmp ult i64 %j, 3
  br i1 %cc, label %outer.latch, label %bail

outer.latch:
  %j.next = add nuw nsw i64 %j, 1
  %oc = icmp eq i64 %j.next, %m
  br i1 %oc, label %done, label %outer.header

bail:
  ret i32 -1

done:
  ret i32 0
}

; AnyOf and FindIV reductions adjust the resume value of the main loop in the
; preheader of the epilogue vector loop.
define i32 @any_of_resume(ptr noalias %src, i64 %n) {
; CHECK-LABEL: VPlan for loop in 'any_of_resume'
; CHECK:  VPlan 'Final VPlan for VF={4},UF={1}' {
; CHECK-NEXT:  Live-in ir<%n> = original trip-count
; CHECK-EMPTY:
; CHECK-NEXT:  ir-bb<entry>:
; CHECK-NEXT:    IR   %min.iters.check = icmp ult i64 %n, 4
; CHECK-NEXT:  Successor(s): ir-bb<vec.epilog.scalar.ph>, ir-bb<vector.main.loop.iter.check>
; CHECK-EMPTY:
; CHECK-NEXT:  ir-bb<vector.main.loop.iter.check>:
; CHECK-NEXT:  Successor(s): vec.epilog.ph, ir-bb<vector.ph>
; CHECK-EMPTY:
; CHECK-NEXT:  ir-bb<vector.ph>:
; CHECK-NEXT:  Successor(s): ir-bb<vector.body>
; CHECK-EMPTY:
; CHECK-NEXT:  ir-bb<vector.body>:
; CHECK-NEXT:  Successor(s): ir-bb<middle.block>
; CHECK-EMPTY:
; CHECK-NEXT:  ir-bb<middle.block>:
; CHECK-NEXT:  Successor(s): ir-bb<vec.epilog.iter.check>
; CHECK-EMPTY:
; CHECK-NEXT:  ir-bb<vec.epilog.iter.check>:
; CHECK-NEXT:    EMIT vp<%min.epilog.iters.check> = icmp ult ir<%0>, ir<4>
; CHECK-NEXT:    EMIT branch-on-cond vp<%min.epilog.iters.check>
; CHECK-NEXT:  Successor(s): ir-bb<vec.epilog.scalar.ph>, vec.epilog.ph
; CHECK-EMPTY:
; CHECK-NEXT:  vec.epilog.ph:
; CHECK-NEXT:    EMIT-SCALAR vp<%vec.epilog.resume.val> = phi [ ir<%n.vec>, ir-bb<vec.epilog.iter.check> ], [ ir<0>, ir-bb<vector.main.loop.iter.check> ]
; CHECK-NEXT:    EMIT-SCALAR vp<%bc.merge.rdx> = phi [ ir<%rdx.select>, ir-bb<vec.epilog.iter.check> ], [ ir<0>, ir-bb<vector.main.loop.iter.check> ]
; CHECK-NEXT:    EMIT vp<[[VP3:%[0-9]+]]> = and ir<%n>, ir<3>
; CHECK-NEXT:    EMIT vp<%n.vec> = sub ir<%n>, vp<[[VP3]]>
; CHECK-NEXT:    EMIT vp<[[VP4:%[0-9]+]]> = icmp ne vp<%bc.merge.rdx>, ir<0>
; CHECK-NEXT:    EMIT vp<[[VP5:%[0-9]+]]> = broadcast vp<[[VP4]]>
; CHECK-NEXT:  Successor(s): vec.epilog.vector.body
; CHECK-EMPTY:
; CHECK-NEXT:  vec.epilog.vector.body:
; CHECK-NEXT:    EMIT-SCALAR vp<%index> = phi [ vp<%vec.epilog.resume.val>, vec.epilog.ph ], [ vp<%index.next>, vec.epilog.vector.body ]
; CHECK-NEXT:    WIDEN-REDUCTION-PHI ir<%rdx> = phi (any-of) vp<[[VP5]]>, vp<[[VP6:%[0-9]+]]>
; CHECK-NEXT:    CLONE ir<%gep> = getelementptr inbounds ir<%src>, vp<%index>
; CHECK-NEXT:    WIDEN ir<%l> = load ir<%gep>
; CHECK-NEXT:    WIDEN ir<%c> = icmp eq ir<%l>, ir<0>
; CHECK-NEXT:    EMIT vp<[[VP6]]> = or ir<%rdx>, ir<%c>
; CHECK-NEXT:    EMIT vp<%index.next> = add nuw vp<%index>, ir<4>
; CHECK-NEXT:    EMIT vp<[[VP7:%[0-9]+]]> = icmp eq vp<%index.next>, vp<%n.vec>
; CHECK-NEXT:    EMIT branch-on-cond vp<[[VP7]]>
; CHECK-NEXT:  Successor(s): vec.epilog.middle.block, vec.epilog.vector.body
; CHECK-EMPTY:
; CHECK-NEXT:  vec.epilog.middle.block:
; CHECK-NEXT:    EMIT vp<[[VP9:%[0-9]+]]> = compute-reduction-result (or) vp<[[VP6]]>
; CHECK-NEXT:    EMIT vp<[[VP10:%[0-9]+]]> = freeze vp<[[VP9]]>
; CHECK-NEXT:    EMIT vp<%rdx.select> = select vp<[[VP10]]>, ir<1>, ir<0>
; CHECK-NEXT:    EMIT vp<%cmp.n> = icmp eq ir<%n>, vp<%n.vec>
; CHECK-NEXT:    EMIT branch-on-cond vp<%cmp.n>
; CHECK-NEXT:  Successor(s): ir-bb<exit>, ir-bb<vec.epilog.scalar.ph>
; CHECK-EMPTY:
; CHECK-NEXT:  ir-bb<exit>:
; CHECK-NEXT:    IR   %sel.lcssa = phi i32 [ %sel, %loop ], [ %rdx.select, %middle.block ] (extra operand: vp<%rdx.select> from vec.epilog.middle.block)
; CHECK-NEXT:  No successors
; CHECK-EMPTY:
; CHECK-NEXT:  ir-bb<vec.epilog.scalar.ph>:
; CHECK-NEXT:    EMIT-SCALAR vp<%bc.resume.val> = phi [ vp<%n.vec>, vec.epilog.middle.block ], [ ir<%n.vec>, ir-bb<vec.epilog.iter.check> ], [ ir<0>, ir-bb<entry> ]
; CHECK-NEXT:    EMIT-SCALAR vp<%bc.merge.rdx>.1 = phi [ vp<%rdx.select>, vec.epilog.middle.block ], [ ir<%rdx.select>, ir-bb<vec.epilog.iter.check> ], [ ir<0>, ir-bb<entry> ]
; CHECK-NEXT:  Successor(s): ir-bb<loop>
; CHECK-EMPTY:
; CHECK-NEXT:  ir-bb<loop>:
; CHECK-NEXT:    IR   %iv = phi i64 [ 0, %vec.epilog.scalar.ph ], [ %iv.next, %loop ] (extra operand: vp<%bc.resume.val> from ir-bb<vec.epilog.scalar.ph>)
; CHECK-NEXT:    IR   %rdx = phi i32 [ 0, %vec.epilog.scalar.ph ], [ %sel, %loop ] (extra operand: vp<%bc.merge.rdx>.1 from ir-bb<vec.epilog.scalar.ph>)
; CHECK-NEXT:    IR   %gep = getelementptr inbounds i8, ptr %src, i64 %iv
; CHECK-NEXT:    IR   %l = load i8, ptr %gep, align 1
; CHECK-NEXT:    IR   %c = icmp eq i8 %l, 0
; CHECK-NEXT:    IR   %sel = select i1 %c, i32 1, i32 %rdx
; CHECK-NEXT:    IR   %iv.next = add i64 %iv, 1
; CHECK-NEXT:    IR   %ec = icmp eq i64 %iv.next, %n
; CHECK-NEXT:  No successors
; CHECK-NEXT:  }
;
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %rdx = phi i32 [ 0, %entry ], [ %sel, %loop ]
  %gep = getelementptr inbounds i8, ptr %src, i64 %iv
  %l = load i8, ptr %gep
  %c = icmp eq i8 %l, 0
  %sel = select i1 %c, i32 1, i32 %rdx
  %iv.next = add i64 %iv, 1
  %ec = icmp eq i64 %iv.next, %n
  br i1 %ec, label %exit, label %loop

exit:
  ret i32 %sel
}

define i64 @find_iv_resume(ptr noalias %a, i64 %n) {
; CHECK-LABEL: VPlan for loop in 'find_iv_resume'
; CHECK:  VPlan 'Final VPlan for VF={4},UF={1}' {
; CHECK-NEXT:  Live-in ir<%n> = original trip-count
; CHECK-EMPTY:
; CHECK-NEXT:  ir-bb<entry>:
; CHECK-NEXT:    IR   %min.iters.check = icmp ult i64 %n, 4
; CHECK-NEXT:  Successor(s): ir-bb<vec.epilog.scalar.ph>, ir-bb<vector.main.loop.iter.check>
; CHECK-EMPTY:
; CHECK-NEXT:  ir-bb<vector.main.loop.iter.check>:
; CHECK-NEXT:  Successor(s): vec.epilog.ph, ir-bb<vector.ph>
; CHECK-EMPTY:
; CHECK-NEXT:  ir-bb<vector.ph>:
; CHECK-NEXT:  Successor(s): ir-bb<vector.body>
; CHECK-EMPTY:
; CHECK-NEXT:  ir-bb<vector.body>:
; CHECK-NEXT:  Successor(s): ir-bb<middle.block>
; CHECK-EMPTY:
; CHECK-NEXT:  ir-bb<middle.block>:
; CHECK-NEXT:  Successor(s): ir-bb<vec.epilog.iter.check>
; CHECK-EMPTY:
; CHECK-NEXT:  ir-bb<vec.epilog.iter.check>:
; CHECK-NEXT:    EMIT vp<%min.epilog.iters.check> = icmp ult ir<%0>, ir<4>
; CHECK-NEXT:    EMIT branch-on-cond vp<%min.epilog.iters.check>
; CHECK-NEXT:  Successor(s): ir-bb<vec.epilog.scalar.ph>, vec.epilog.ph
; CHECK-EMPTY:
; CHECK-NEXT:  vec.epilog.ph:
; CHECK-NEXT:    EMIT-SCALAR vp<%vec.epilog.resume.val> = phi [ ir<%n.vec>, ir-bb<vec.epilog.iter.check> ], [ ir<0>, ir-bb<vector.main.loop.iter.check> ]
; CHECK-NEXT:    EMIT-SCALAR vp<%bc.merge.rdx> = phi [ ir<%7>, ir-bb<vec.epilog.iter.check> ], [ ir<3>, ir-bb<vector.main.loop.iter.check> ]
; CHECK-NEXT:    EMIT vp<[[VP3:%[0-9]+]]> = and ir<%n>, ir<3>
; CHECK-NEXT:    EMIT vp<%n.vec> = sub ir<%n>, vp<[[VP3]]>
; CHECK-NEXT:    EMIT vp<[[VP4:%[0-9]+]]> = icmp eq vp<%bc.merge.rdx>, ir<3>
; CHECK-NEXT:    EMIT vp<[[VP5:%[0-9]+]]> = select vp<[[VP4]]>, ir<-9223372036854775808>, vp<%bc.merge.rdx>
; CHECK-NEXT:    EMIT vp<[[VP6:%[0-9]+]]> = broadcast vp<[[VP5]]>
; CHECK-NEXT:    EMIT vp<[[VP7:%[0-9]+]]> = step-vector i64
; CHECK-NEXT:    EMIT vp<[[VP8:%[0-9]+]]> = broadcast vp<%vec.epilog.resume.val>
; CHECK-NEXT:    EMIT vp<%induction> = add nuw nsw vp<[[VP8]]>, vp<[[VP7]]>
; CHECK-NEXT:    EMIT vp<[[VP9:%[0-9]+]]> = broadcast ir<4>
; CHECK-NEXT:  Successor(s): vec.epilog.vector.body
; CHECK-EMPTY:
; CHECK-NEXT:  vec.epilog.vector.body:
; CHECK-NEXT:    EMIT-SCALAR vp<%index> = phi [ vp<%vec.epilog.resume.val>, vec.epilog.ph ], [ vp<%index.next>, vec.epilog.vector.body ]
; CHECK-NEXT:    WIDEN-PHI vp<[[VP10:%[0-9]+]]> = phi [ vp<%induction>, vec.epilog.ph ], [ vp<%vec.ind.next>, vec.epilog.vector.body ]
; CHECK-NEXT:    WIDEN-REDUCTION-PHI ir<%rdx> = phi (find-iv) vp<[[VP6]]>, ir<%sel>
; CHECK-NEXT:    CLONE ir<%gep> = getelementptr inbounds ir<%a>, vp<%index>
; CHECK-NEXT:    WIDEN ir<%l> = load ir<%gep>
; CHECK-NEXT:    WIDEN ir<%c> = icmp eq ir<%l>, ir<3>
; CHECK-NEXT:    WIDEN ir<%sel> = select ir<%c>, vp<[[VP10]]>, ir<%rdx>
; CHECK-NEXT:    EMIT vp<%index.next> = add nuw vp<%index>, ir<4>
; CHECK-NEXT:    EMIT vp<%vec.ind.next> = add nuw nsw vp<[[VP10]]>, vp<[[VP9]]>
; CHECK-NEXT:    EMIT vp<[[VP11:%[0-9]+]]> = icmp eq vp<%index.next>, vp<%n.vec>
; CHECK-NEXT:    EMIT branch-on-cond vp<[[VP11]]>
; CHECK-NEXT:  Successor(s): vec.epilog.middle.block, vec.epilog.vector.body
; CHECK-EMPTY:
; CHECK-NEXT:  vec.epilog.middle.block:
; CHECK-NEXT:    EMIT vp<[[VP13:%[0-9]+]]> = compute-reduction-result (smax) ir<%sel>
; CHECK-NEXT:    EMIT vp<[[VP14:%[0-9]+]]> = icmp ne vp<[[VP13]]>, ir<-9223372036854775808>
; CHECK-NEXT:    EMIT vp<[[VP15:%[0-9]+]]> = select vp<[[VP14]]>, vp<[[VP13]]>, ir<3>
; CHECK-NEXT:    EMIT vp<%cmp.n> = icmp eq ir<%n>, vp<%n.vec>
; CHECK-NEXT:    EMIT branch-on-cond vp<%cmp.n>
; CHECK-NEXT:  Successor(s): ir-bb<exit>, ir-bb<vec.epilog.scalar.ph>
; CHECK-EMPTY:
; CHECK-NEXT:  ir-bb<exit>:
; CHECK-NEXT:    IR   %sel.lcssa = phi i64 [ %sel, %loop ], [ %7, %middle.block ] (extra operand: vp<[[VP15]]> from vec.epilog.middle.block)
; CHECK-NEXT:  No successors
; CHECK-EMPTY:
; CHECK-NEXT:  ir-bb<vec.epilog.scalar.ph>:
; CHECK-NEXT:    EMIT-SCALAR vp<%bc.resume.val> = phi [ vp<%n.vec>, vec.epilog.middle.block ], [ ir<%n.vec>, ir-bb<vec.epilog.iter.check> ], [ ir<0>, ir-bb<entry> ]
; CHECK-NEXT:    EMIT-SCALAR vp<%bc.merge.rdx>.1 = phi [ vp<[[VP15]]>, vec.epilog.middle.block ], [ ir<%7>, ir-bb<vec.epilog.iter.check> ], [ ir<3>, ir-bb<entry> ]
; CHECK-NEXT:  Successor(s): ir-bb<loop>
; CHECK-EMPTY:
; CHECK-NEXT:  ir-bb<loop>:
; CHECK-NEXT:    IR   %iv = phi i64 [ 0, %vec.epilog.scalar.ph ], [ %iv.next, %loop ] (extra operand: vp<%bc.resume.val> from ir-bb<vec.epilog.scalar.ph>)
; CHECK-NEXT:    IR   %rdx = phi i64 [ 3, %vec.epilog.scalar.ph ], [ %sel, %loop ] (extra operand: vp<%bc.merge.rdx>.1 from ir-bb<vec.epilog.scalar.ph>)
; CHECK-NEXT:    IR   %gep = getelementptr inbounds i64, ptr %a, i64 %iv
; CHECK-NEXT:    IR   %l = load i64, ptr %gep, align 4
; CHECK-NEXT:    IR   %c = icmp eq i64 %l, 3
; CHECK-NEXT:    IR   %sel = select i1 %c, i64 %iv, i64 %rdx
; CHECK-NEXT:    IR   %iv.next = add nuw nsw i64 %iv, 1
; CHECK-NEXT:    IR   %ec = icmp eq i64 %iv.next, %n
; CHECK-NEXT:  No successors
; CHECK-NEXT:  }
;
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %rdx = phi i64 [ 3, %entry ], [ %sel, %loop ]
  %gep = getelementptr inbounds i64, ptr %a, i64 %iv
  %l = load i64, ptr %gep
  %c = icmp eq i64 %l, 3
  %sel = select i1 %c, i64 %iv, i64 %rdx
  %iv.next = add nuw nsw i64 %iv, 1
  %ec = icmp eq i64 %iv.next, %n
  br i1 %ec, label %exit, label %loop

exit:
  ret i64 %sel
}
