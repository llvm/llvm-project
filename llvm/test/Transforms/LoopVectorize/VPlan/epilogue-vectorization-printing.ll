; RUN: opt -passes=loop-vectorize -force-vector-width=8 -enable-epilogue-vectorization \
; RUN:     -epilogue-vectorization-force-VF=4 -vplan-print-metadata=false -disable-output \
; RUN:     -vplan-print-after=printFinalVPlan %s 2>&1 | FileCheck %s

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
; CHECK-NEXT:    EMIT vp<[[VP8:%[0-9]+]]> = compute-reduction-result (add) ir<%red.next>
; CHECK-NEXT:    EMIT vp<%cmp.n> = icmp eq ir<%n>, vp<%n.vec>
; CHECK-NEXT:    EMIT branch-on-cond vp<%cmp.n>
; CHECK-NEXT:  Successor(s): ir-bb<exit>, ir-bb<scalar.ph>
; CHECK-EMPTY:
; CHECK-NEXT:  ir-bb<exit>:
; CHECK-NEXT:    IR   %red.next.lcssa = phi i64 [ %red.next, %loop ] (extra operand: vp<[[VP8]]> from middle.block)
; CHECK-NEXT:  No successors
; CHECK-EMPTY:
; CHECK-NEXT:  ir-bb<scalar.ph>:
; CHECK-NEXT:    EMIT-SCALAR vp<%vec.epilog.resume.val> = phi [ vp<%n.vec>, middle.block ], [ ir<0>, ir-bb<entry> ], [ ir<0>, vector.main.loop.iter.check ]
; CHECK-NEXT:    EMIT-SCALAR vp<%bc.merge.rdx> = phi [ vp<[[VP8]]>, middle.block ], [ ir<5>, ir-bb<entry> ], [ ir<5>, vector.main.loop.iter.check ]
; CHECK-NEXT:    EMIT-SCALAR vp<[[VP10:%[0-9]+]]> = resume-for-epilogue vp<%vec.epilog.resume.val>, ir<0>
; CHECK-NEXT:    EMIT-SCALAR vp<[[VP11:%[0-9]+]]> = resume-for-epilogue vp<%vec.epilog.resume.val>, ir<0>
; CHECK-NEXT:    EMIT-SCALAR vp<[[VP12:%[0-9]+]]> = resume-for-epilogue vp<%bc.merge.rdx>, ir<5>
; CHECK-NEXT:  Successor(s): ir-bb<loop>
; CHECK-EMPTY:
; CHECK-NEXT:  ir-bb<loop>:
; CHECK-NEXT:    IR   %iv = phi i64 [ 0, %scalar.ph ], [ %iv.next, %loop ] (extra operand: vp<%vec.epilog.resume.val> from ir-bb<scalar.ph>)
; CHECK-NEXT:    IR   %red = phi i64 [ 5, %scalar.ph ], [ %red.next, %loop ] (extra operand: vp<%bc.merge.rdx> from ir-bb<scalar.ph>)
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
; CHECK-NEXT:  ir-bb<vec.epilog.iter.check>:
; CHECK-NEXT:    IR   %vec.epilog.resume.val = phi i64 [ %n.vec, %middle.block ], [ 0, %iter.check ], [ 0, %vector.main.loop.iter.check ]
; CHECK-NEXT:    IR   %bc.merge.rdx = phi i64 [ %4, %middle.block ], [ 5, %iter.check ], [ 5, %vector.main.loop.iter.check ]
; CHECK-NEXT:    EMIT vp<%min.epilog.iters.check> = icmp ult ir<%0>, ir<4>
; CHECK-NEXT:    EMIT branch-on-cond vp<%min.epilog.iters.check>
; CHECK-NEXT:  Successor(s): ir-bb<vec.epilog.scalar.ph>, vec.epilog.ph
; CHECK-EMPTY:
; CHECK-NEXT:  vec.epilog.ph:
; CHECK-NEXT:    EMIT vp<[[VP3:%[0-9]+]]> = and ir<%n>, ir<3>
; CHECK-NEXT:    EMIT vp<%n.vec> = sub ir<%n>, vp<[[VP3]]>
; CHECK-NEXT:    EMIT vp<[[VP4:%[0-9]+]]> = reduction-start-vector ir<%bc.merge.rdx>, ir<0>, ir<1>
; CHECK-NEXT:  Successor(s): vec.epilog.vector.body
; CHECK-EMPTY:
; CHECK-NEXT:  vec.epilog.vector.body:
; CHECK-NEXT:    EMIT-SCALAR vp<%index> = phi [ ir<%vec.epilog.resume.val>, vec.epilog.ph ], [ vp<%index.next>, vec.epilog.vector.body ]
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
; CHECK-NEXT:    EMIT-SCALAR vp<%bc.resume.val> = phi [ vp<%n.vec>, vec.epilog.middle.block ], [ ir<0>, ir-bb<vec.epilog.iter.check> ]
; CHECK-NEXT:    EMIT-SCALAR vp<%bc.merge.rdx> = phi [ vp<[[VP7]]>, vec.epilog.middle.block ], [ ir<5>, ir-bb<vec.epilog.iter.check> ]
; CHECK-NEXT:  Successor(s): ir-bb<loop>
; CHECK-EMPTY:
; CHECK-NEXT:  ir-bb<loop>:
; CHECK-NEXT:    IR   %iv = phi i64 [ %vec.epilog.resume.val, %vec.epilog.scalar.ph ], [ %iv.next, %loop ] (extra operand: vp<%bc.resume.val> from ir-bb<vec.epilog.scalar.ph>)
; CHECK-NEXT:    IR   %red = phi i64 [ %bc.merge.rdx, %vec.epilog.scalar.ph ], [ %red.next, %loop ] (extra operand: vp<%bc.merge.rdx> from ir-bb<vec.epilog.scalar.ph>)
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
; CHECK-NEXT:    EMIT vp<[[VP10:%[0-9]+]]> = compute-reduction-result (add) ir<%red.next>
; CHECK-NEXT:    EMIT vp<%cmp.n> = icmp eq ir<%n>, vp<%n.vec>
; CHECK-NEXT:    EMIT branch-on-cond vp<%cmp.n>
; CHECK-NEXT:  Successor(s): ir-bb<exit>, ir-bb<scalar.ph>
; CHECK-EMPTY:
; CHECK-NEXT:  ir-bb<exit>:
; CHECK-NEXT:    IR   %red.next.lcssa = phi i64 [ %red.next, %loop ] (extra operand: vp<[[VP10]]> from middle.block)
; CHECK-NEXT:  No successors
; CHECK-EMPTY:
; CHECK-NEXT:  ir-bb<scalar.ph>:
; CHECK-NEXT:    EMIT-SCALAR vp<%vec.epilog.resume.val> = phi [ vp<%n.vec>, middle.block ], [ ir<0>, ir-bb<entry> ], [ ir<0>, ir-bb<vector.scevcheck> ], [ ir<0>, ir-bb<vector.memcheck> ], [ ir<0>, vector.main.loop.iter.check ]
; CHECK-NEXT:    EMIT-SCALAR vp<%bc.merge.rdx> = phi [ vp<[[VP10]]>, middle.block ], [ ir<0>, ir-bb<entry> ], [ ir<0>, ir-bb<vector.scevcheck> ], [ ir<0>, ir-bb<vector.memcheck> ], [ ir<0>, vector.main.loop.iter.check ]
; CHECK-NEXT:    EMIT-SCALAR vp<[[VP12:%[0-9]+]]> = resume-for-epilogue vp<%vec.epilog.resume.val>, ir<0>
; CHECK-NEXT:    EMIT-SCALAR vp<[[VP13:%[0-9]+]]> = resume-for-epilogue vp<%vec.epilog.resume.val>, ir<0>
; CHECK-NEXT:    EMIT-SCALAR vp<[[VP14:%[0-9]+]]> = resume-for-epilogue vp<%bc.merge.rdx>, ir<0>
; CHECK-NEXT:  Successor(s): ir-bb<loop>
; CHECK-EMPTY:
; CHECK-NEXT:  ir-bb<loop>:
; CHECK-NEXT:    IR   %iv = phi i32 [ 0, %scalar.ph ], [ %iv.next, %loop ] (extra operand: vp<%vec.epilog.resume.val> from ir-bb<scalar.ph>)
; CHECK-NEXT:    IR   %red = phi i64 [ 0, %scalar.ph ], [ %red.next, %loop ] (extra operand: vp<%bc.merge.rdx> from ir-bb<scalar.ph>)
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
; CHECK-NEXT:  ir-bb<vec.epilog.iter.check>:
; CHECK-NEXT:    IR   %vec.epilog.resume.val = phi i32 [ %n.vec, %middle.block ], [ 0, %iter.check ], [ 0, %vector.scevcheck ], [ 0, %vector.memcheck ], [ 0, %vector.main.loop.iter.check ]
; CHECK-NEXT:    IR   %bc.merge.rdx = phi i64 [ %10, %middle.block ], [ 0, %iter.check ], [ 0, %vector.scevcheck ], [ 0, %vector.memcheck ], [ 0, %vector.main.loop.iter.check ]
; CHECK-NEXT:    EMIT vp<%min.epilog.iters.check> = icmp ult ir<%4>, ir<4>
; CHECK-NEXT:    EMIT branch-on-cond vp<%min.epilog.iters.check>
; CHECK-NEXT:  Successor(s): ir-bb<vec.epilog.scalar.ph>, vec.epilog.ph
; CHECK-EMPTY:
; CHECK-NEXT:  vec.epilog.ph:
; CHECK-NEXT:    EMIT vp<[[VP3:%[0-9]+]]> = and ir<%n>, ir<3>
; CHECK-NEXT:    EMIT vp<%n.vec> = sub ir<%n>, vp<[[VP3]]>
; CHECK-NEXT:    EMIT vp<[[VP4:%[0-9]+]]> = reduction-start-vector ir<%bc.merge.rdx>, ir<0>, ir<1>
; CHECK-NEXT:  Successor(s): vec.epilog.vector.body
; CHECK-EMPTY:
; CHECK-NEXT:  vec.epilog.vector.body:
; CHECK-NEXT:    EMIT-SCALAR vp<%index> = phi [ ir<%vec.epilog.resume.val>, vec.epilog.ph ], [ vp<%index.next>, vec.epilog.vector.body ]
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
; CHECK-NEXT:    EMIT-SCALAR vp<%bc.resume.val> = phi [ vp<%n.vec>, vec.epilog.middle.block ], [ ir<0>, ir-bb<vec.epilog.iter.check> ]
; CHECK-NEXT:    EMIT-SCALAR vp<%bc.merge.rdx> = phi [ vp<[[VP7]]>, vec.epilog.middle.block ], [ ir<0>, ir-bb<vec.epilog.iter.check> ]
; CHECK-NEXT:  Successor(s): ir-bb<loop>
; CHECK-EMPTY:
; CHECK-NEXT:  ir-bb<loop>:
; CHECK-NEXT:    IR   %iv = phi i32 [ %vec.epilog.resume.val, %vec.epilog.scalar.ph ], [ %iv.next, %loop ] (extra operand: vp<%bc.resume.val> from ir-bb<vec.epilog.scalar.ph>)
; CHECK-NEXT:    IR   %red = phi i64 [ %bc.merge.rdx, %vec.epilog.scalar.ph ], [ %red.next, %loop ] (extra operand: vp<%bc.merge.rdx> from ir-bb<vec.epilog.scalar.ph>)
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
