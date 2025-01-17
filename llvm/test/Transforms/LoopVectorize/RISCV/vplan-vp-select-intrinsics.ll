; REQUIRES: asserts

 ; RUN: opt -passes=loop-vectorize -debug-only=loop-vectorize \
 ; RUN: -force-tail-folding-style=data-with-evl \
 ; RUN: -prefer-predicate-over-epilogue=predicate-dont-vectorize \
 ; RUN: -mtriple=riscv64 -mattr=+v -riscv-v-vector-bits-max=128 -disable-output < %s 2>&1 | FileCheck --check-prefix=IF-EVL %s

 define void @vp_select(ptr noalias %a, ptr noalias %b, ptr noalias %c, i64 %N) {
 ; IF-EVL: VPlan 'Final VPlan for VF={vscale x 1,vscale x 2,vscale x 4},UF={1}' {
 ; IF-EVL-NEXT: Live-in ir<[[VFUF:%.+]]> = VF * UF
 ; IF-EVL-NEXT: Live-in ir<[[VTC:%.+]]> = vector-trip-count
 ; IF-EVL-NEXT: Live-in ir<%N> = original trip-count

 ; IF-EVL:      ir-bb<entry>:
 ; IF-EVL-NEXT: Successor(s): ir-bb<scalar.ph>, ir-bb<vector.ph>

 ; IF-EVL:      ir-bb<vector.ph>:
 ; IF-EVL-NEXT:   IR   %4 = call i64 @llvm.vscale.i64()
 ; IF-EVL-NEXT:   IR   %5 = mul i64 %4, 4
 ; IF-EVL-NEXT:   IR   %6 = sub i64 %5, 1
 ; IF-EVL-NEXT:   IR   %n.rnd.up = add i64 %N, %6
 ; IF-EVL-NEXT:   IR   %n.mod.vf = urem i64 %n.rnd.up, %5
 ; IF-EVL-NEXT:   IR   %n.vec = sub i64 %n.rnd.up, %n.mod.vf
 ; IF-EVL-NEXT:   IR   %7 = call i64 @llvm.vscale.i64()
 ; IF-EVL-NEXT:   IR   %8 = mul i64 %7, 4
 ; IF-EVL-NEXT: Successor(s): vector loop

 ; IF-EVL: <x1> vector loop: {
 ; IF-EVL-NEXT:   vector.body:
 ; IF-EVL-NEXT:     SCALAR-PHI vp<[[IV:%[0-9]+]]> = phi ir<0>, vp<[[IV_NEXT_EXIT:%.+]]>
 ; IF-EVL-NEXT:     SCALAR-PHI vp<[[EVL_PHI:%[0-9]+]]>  = phi ir<0>, vp<[[IV_NEX:%.+]]>
 ; IF-EVL-NEXT:     EMIT vp<[[AVL:%.+]]> = sub ir<%N>, vp<[[EVL_PHI]]>
 ; IF-EVL-NEXT:     EMIT vp<[[EVL:%.+]]> = EXPLICIT-VECTOR-LENGTH vp<[[AVL]]>
 ; IF-EVL-NEXT:     vp<[[ST:%[0-9]+]]> = SCALAR-STEPS vp<[[EVL_PHI]]>, ir<1>
 ; IF-EVL-NEXT:     CLONE ir<[[GEP1:%.+]]> = getelementptr inbounds ir<%b>, vp<[[ST]]>
 ; IF-EVL-NEXT:     vp<[[PTR1:%[0-9]+]]> = vector-pointer ir<[[GEP1]]>
 ; IF-EVL-NEXT:     WIDEN ir<[[LD1:%.+]]> = vp.load vp<[[PTR1]]>, vp<[[EVL]]>
 ; IF-EVL-NEXT:     CLONE ir<[[GEP2:%.+]]> = getelementptr inbounds ir<%c>, vp<[[ST]]>
 ; IF-EVL-NEXT:     vp<[[PTR2:%[0-9]+]]> = vector-pointer ir<[[GEP2]]>
 ; IF-EVL-NEXT:     WIDEN ir<[[LD2:%.+]]> = vp.load vp<[[PTR2]]>, vp<[[EVL]]>
 ; IF-EVL-NEXT:     WIDEN ir<[[CMP:%.+]]> = icmp sgt ir<[[LD1]]>, ir<[[LD2]]>
 ; IF-EVL-NEXT:     WIDEN ir<[[SUB:%.+]]> = vp.sub ir<0>, ir<[[LD2]]>, vp<[[EVL]]>
 ; IF-EVL-NEXT:     WIDEN-INTRINSIC vp<[[SELECT:%.+]]> = call llvm.vp.select(ir<[[CMP]]>, ir<[[LD2]]>, ir<[[SUB]]>, vp<[[EVL]]>)
 ; IF-EVL-NEXT:     WIDEN ir<[[ADD:%.+]]> = vp.add vp<[[SELECT]]>, ir<[[LD1]]>, vp<[[EVL]]>
 ; IF-EVL-NEXT:     CLONE ir<[[GEP3:%.+]]> = getelementptr inbounds ir<%a>, vp<[[ST]]>
 ; IF-EVL-NEXT:     vp<[[PTR3:%.+]]> = vector-pointer ir<[[GEP3]]>
 ; IF-EVL-NEXT:     WIDEN vp.store vp<[[PTR3]]>, ir<[[ADD]]>, vp<[[EVL]]>
 ; IF-EVL-NEXT:     SCALAR-CAST vp<[[CAST:%[0-9]+]]> = zext vp<[[EVL]]> to i64
 ; IF-EVL-NEXT:     EMIT vp<[[IV_NEX]]> = add vp<[[CAST]]>, vp<[[EVL_PHI]]>
 ; IF-EVL-NEXT:     EMIT vp<[[IV_NEXT_EXIT]]> = add vp<[[IV]]>, ir<[[VFUF]]>
 ; IF-EVL-NEXT:     EMIT branch-on-count vp<[[IV_NEXT_EXIT]]>,  ir<[[VTC]]>
 ; IF-EVL-NEXT:   No successors
 ; IF-EVL-NEXT: }

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
