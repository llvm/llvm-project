; RUN: opt < %s -S -passes='loop(loop-flatten),verify' \  
; RUN:   -verify-loop-info -verify-dom-info -verify-scev | FileCheck %s  
  
; An outer-only load may be executed once per inner iteration after  
; flattening, so it must produce the same value every time. A load that  
; passes isSafeToSpeculativelyExecute() can still observe the inner  
; loop's stores, so checkOuterLoopInsts() must reject outer-only  
; instructions that mayReadFromMemory() when the nest may write.  
  
@g = global i32 7, align 4  
@c = constant i32 11, align 4  
  
; The original miscompile: the inner loop stores to @g, so the outer-only  
; load of @g returns a different value after flattening.  
define i32 @main() {  
; CHECK-LABEL: @main(  
; CHECK-NOT:   flatten.tripcount    
; CHECK:       %x = load i32, ptr @g
; CHECK:       %j = phi i32 [ 0, %outer ], [ %j.next, %inner.latch ]  
; CHECK:       ret i32 %x.lcssa  
;
entry:  
  br label %outer  
  
outer:  
  %i = phi i32 [ 0, %entry ], [ %i.next, %outer.latch ]  
  %x = load i32, ptr @g, align 4  
  br label %inner  
  
inner:  
  %j = phi i32 [ 0, %outer ], [ %j.next, %inner.latch ]  
  store i32 0, ptr @g, align 4  
  br label %inner.latch  
  
inner.latch:  
  %j.next = add i32 %j, 1  
  %j.cmp = icmp ult i32 %j.next, 2  
  br i1 %j.cmp, label %inner, label %outer.latch  
  
outer.latch:  
  %i.next = add i32 %i, 1  
  %i.cmp = icmp ult i32 %i.next, 1  
  br i1 %i.cmp, label %outer, label %exit  
  
exit:  
  ret i32 %x  
}  
  
; Positive control: same linearized-GEP nest with no outer-only load.  
; This must still flatten. 
; 
define void @control(ptr %A) {  
; CHECK-LABEL: @control(  
; CHECK:       flatten.tripcount  
entry:  
  br label %for.inner.preheader  
  
for.inner.preheader:  
  %i = phi i32 [ 0, %entry ], [ %inc2, %for.outer ]  
  br label %for.inner  
  
for.inner:  
  %j = phi i32 [ 0, %for.inner.preheader ], [ %inc1, %for.inner ]  
  %mul = mul i32 %i, 2  
  %arrayidx = getelementptr inbounds i32, ptr %A, i32 %mul  
  %addr = getelementptr inbounds i32, ptr %arrayidx, i32 %j  
  store i32 0, ptr %addr, align 4  
  %inc1 = add nuw i32 %j, 1  
  %cmp2 = icmp ult i32 %inc1, 2  
  br i1 %cmp2, label %for.inner, label %for.outer  
  
for.outer:  
  %inc2 = add i32 %i, 1  
  %cmp1 = icmp ult i32 %inc2, 3  
  br i1 %cmp1, label %for.inner.preheader, label %for.end  
  
for.end:  
  ret void  
}
  
; Outer-only load of @g, disjoint from the inner loop's stores to %A.  
; Still rejected by a blanket mayReadFromMemory() check. 
; 
define void @outer_load_disjoint_global(i32 %N, ptr %A) {  
; CHECK-LABEL: @outer_load_disjoint_global(  
; CHECK-NOT:   flatten.tripcount  
; CHECK:       %j = phi i32 [ 0, %for.inner.preheader ], [ %inc1, %for.inner ]  
; CHECK:       ret void 
; 
entry:  
  %cmp3 = icmp ult i32 0, %N  
  br i1 %cmp3, label %for.outer.preheader, label %for.end  
  
for.outer.preheader:  
  br label %for.inner.preheader  
  
for.inner.preheader:  
  %i = phi i32 [ 0, %for.outer.preheader ], [ %inc2, %for.outer ]  
  %x = load i32, ptr @g, align 4  
  br label %for.inner  
  
for.inner:  
  %j = phi i32 [ 0, %for.inner.preheader ], [ %inc1, %for.inner ]  
  %mul = mul i32 %i, %N  
  %gep = getelementptr inbounds i32, ptr %A, i32 %mul  
  %arrayidx = getelementptr inbounds i32, ptr %gep, i32 %j  
  store i32 %x, ptr %arrayidx, align 4  
  %inc1 = add nuw i32 %j, 1  
  %cmp2 = icmp ult i32 %inc1, %N  
  br i1 %cmp2, label %for.inner, label %for.outer  
  
for.outer:  
  %inc2 = add i32 %i, 1  
  %cmp1 = icmp ult i32 %inc2, %N  
  br i1 %cmp1, label %for.inner.preheader, label %for.end.loopexit  
  
for.end.loopexit:  
  br label %for.end  
  
for.end:  
  ret void  
}  
  
; Same, but loading a constant global: mayReadFromMemory() is still true,  
; so this is also rejected conservatively.
;  
define void @outer_load_constant_disjoint_global(i32 %N, ptr %A) {  
; CHECK-LABEL: @outer_load_constant_disjoint_global(  
; CHECK-NOT:   flatten.tripcount  
; CHECK:       %j = phi i32 [ 0, %for.inner.preheader ], [ %inc1, %for.inner ]  
; CHECK:       ret void
;  
entry:  
  %cmp3 = icmp ult i32 0, %N  
  br i1 %cmp3, label %for.outer.preheader, label %for.end  
  
for.outer.preheader:  
  br label %for.inner.preheader  
  
for.inner.preheader:  
  %i = phi i32 [ 0, %for.outer.preheader ], [ %inc2, %for.outer ]  
  %x = load i32, ptr @c, align 4  
  br label %for.inner  
  
for.inner:  
  %j = phi i32 [ 0, %for.inner.preheader ], [ %inc1, %for.inner ]  
  %mul = mul i32 %i, %N  
  %gep = getelementptr inbounds i32, ptr %A, i32 %mul  
  %arrayidx = getelementptr inbounds i32, ptr %gep, i32 %j  
  store i32 %x, ptr %arrayidx, align 4  
  %inc1 = add nuw i32 %j, 1  
  %cmp2 = icmp ult i32 %inc1, %N  
  br i1 %cmp2, label %for.inner, label %for.outer  
  
for.outer:  
  %inc2 = add i32 %i, 1  
  %cmp1 = icmp ult i32 %inc2, %N  
  br i1 %cmp1, label %for.inner.preheader, label %for.end.loopexit  
  
for.end.loopexit:  
  br label %for.end  
  
for.end:  
  ret void  
}
