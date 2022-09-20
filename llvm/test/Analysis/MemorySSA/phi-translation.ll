; RUN: opt -aa-pipeline=basic-aa -passes='print<memoryssa>,verify<memoryssa>' -disable-output < %s 2>&1 | FileCheck %s --check-prefixes=CHECK,NOLIMIT
; RUN: opt -memssa-check-limit=0 -aa-pipeline=basic-aa -passes='print<memoryssa>,verify<memoryssa>' -disable-output < %s 2>&1 | FileCheck %s --check-prefixes=CHECK,LIMIT

; %ptr can't alias %local, so we should be able to optimize the use of %local to
; point to the store to %local.
; CHECK-LABEL: define void @check
define void @check(i8* %ptr, i1 %bool) {
entry:
  %local = alloca i8, align 1
; CHECK: 1 = MemoryDef(liveOnEntry)
; CHECK-NEXT: store i8 0, i8* %local, align 1
  store i8 0, i8* %local, align 1
  br i1 %bool, label %if.then, label %if.end

if.then:
  %p2 = getelementptr inbounds i8, i8* %ptr, i32 1
; CHECK: 2 = MemoryDef(1)
; CHECK-NEXT: store i8 0, i8* %p2, align 1
  store i8 0, i8* %p2, align 1
  br label %if.end

if.end:
; CHECK: 3 = MemoryPhi({entry,1},{if.then,2})
; NOLIMIT: MemoryUse(1)
; NOLIMIT-NEXT: load i8, i8* %local, align 1
; LIMIT: MemoryUse(3)
; LIMIT-NEXT: load i8, i8* %local, align 1
  load i8, i8* %local, align 1
  ret void
}

; CHECK-LABEL: define void @check2
define void @check2(i1 %val1, i1 %val2, i1 %val3) {
entry:
  %local = alloca i8, align 1
  %local2 = alloca i8, align 1

; CHECK: 1 = MemoryDef(liveOnEntry)
; CHECK-NEXT: store i8 0, i8* %local
  store i8 0, i8* %local
  br i1 %val1, label %if.then, label %phi.3

if.then:
; CHECK: 2 = MemoryDef(1)
; CHECK-NEXT: store i8 2, i8* %local2
  store i8 2, i8* %local2
  br i1 %val2, label %phi.2, label %phi.3

phi.3:
; CHECK: 7 = MemoryPhi({entry,1},{if.then,2})
; CHECK: 3 = MemoryDef(7)
; CHECK-NEXT: store i8 3, i8* %local2
  store i8 3, i8* %local2
  br i1 %val3, label %phi.2, label %phi.1

phi.2:
; CHECK: 5 = MemoryPhi({if.then,2},{phi.3,3})
; CHECK: 4 = MemoryDef(5)
; CHECK-NEXT: store i8 4, i8* %local2
  store i8 4, i8* %local2
  br label %phi.1

phi.1:
; Order matters here; phi.2 needs to come before phi.3, because that's the order
; they're visited in.
; CHECK: 6 = MemoryPhi({phi.2,4},{phi.3,3})
; NOLIMIT: MemoryUse(1)
; NOLIMIT-NEXT: load i8, i8* %local
; LIMIT: MemoryUse(6)
; LIMIT-NEXT: load i8, i8* %local
  load i8, i8* %local
  ret void
}

; CHECK-LABEL: define void @cross_phi
define void @cross_phi(i8* noalias %p1, i8* noalias %p2) {
; CHECK: 1 = MemoryDef(liveOnEntry)
; CHECK-NEXT: store i8 0, i8* %p1
  store i8 0, i8* %p1
; NOLIMIT: MemoryUse(1)
; NOLIMIT-NEXT: load i8, i8* %p1
; LIMIT: MemoryUse(1)
; LIMIT-NEXT: load i8, i8* %p1
  load i8, i8* %p1
  br i1 undef, label %a, label %b

a:
; CHECK: 2 = MemoryDef(1)
; CHECK-NEXT: store i8 0, i8* %p2
  store i8 0, i8* %p2
  br i1 undef, label %c, label %d

b:
; CHECK: 3 = MemoryDef(1)
; CHECK-NEXT: store i8 1, i8* %p2
  store i8 1, i8* %p2
  br i1 undef, label %c, label %d

c:
; CHECK: 6 = MemoryPhi({a,2},{b,3})
; CHECK: 4 = MemoryDef(6)
; CHECK-NEXT: store i8 2, i8* %p2
  store i8 2, i8* %p2
  br label %e

d:
; CHECK: 7 = MemoryPhi({a,2},{b,3})
; CHECK: 5 = MemoryDef(7)
; CHECK-NEXT: store i8 3, i8* %p2
  store i8 3, i8* %p2
  br label %e

e:
; 8 = MemoryPhi({c,4},{d,5})
; NOLIMIT: MemoryUse(1)
; NOLIMIT-NEXT: load i8, i8* %p1
; LIMIT: MemoryUse(8)
; LIMIT-NEXT: load i8, i8* %p1
  load i8, i8* %p1
  ret void
}

; CHECK-LABEL: define void @looped
define void @looped(i8* noalias %p1, i8* noalias %p2) {
; CHECK: 1 = MemoryDef(liveOnEntry)
; CHECK-NEXT: store i8 0, i8* %p1
  store i8 0, i8* %p1
  br label %loop.1

loop.1:
; CHECK: 6 = MemoryPhi({%0,1},{loop.3,4})
; CHECK: 2 = MemoryDef(6)
; CHECK-NEXT: store i8 0, i8* %p2
  store i8 0, i8* %p2
  br i1 undef, label %loop.2, label %loop.3

loop.2:
; CHECK: 5 = MemoryPhi({loop.1,2},{loop.3,4})
; CHECK: 3 = MemoryDef(5)
; CHECK-NEXT: store i8 1, i8* %p2
  store i8 1, i8* %p2
  br label %loop.3

loop.3:
; CHECK: 7 = MemoryPhi({loop.1,2},{loop.2,3})
; CHECK: 4 = MemoryDef(7)
; CHECK-NEXT: store i8 2, i8* %p2
  store i8 2, i8* %p2
; NOLIMIT: MemoryUse(1)
; NOLIMIT-NEXT: load i8, i8* %p1
; LIMIT: MemoryUse(4)
; LIMIT-NEXT: load i8, i8* %p1
  load i8, i8* %p1
  br i1 undef, label %loop.2, label %loop.1
}

; CHECK-LABEL: define void @looped_visitedonlyonce
define void @looped_visitedonlyonce(i8* noalias %p1, i8* noalias %p2) {
  br label %while.cond

while.cond:
; CHECK: 5 = MemoryPhi({%0,liveOnEntry},{if.end,3})
; CHECK-NEXT: br i1 undef, label %if.then, label %if.end
  br i1 undef, label %if.then, label %if.end

if.then:
; CHECK: 1 = MemoryDef(5)
; CHECK-NEXT: store i8 0, i8* %p1
  store i8 0, i8* %p1
  br i1 undef, label %if.end, label %if.then2

if.then2:
; CHECK: 2 = MemoryDef(1)
; CHECK-NEXT: store i8 1, i8* %p2
  store i8 1, i8* %p2
  br label %if.end

if.end:
; CHECK: 4 = MemoryPhi({while.cond,5},{if.then,1},{if.then2,2})
; CHECK: MemoryUse(4)
; CHECK-NEXT: load i8, i8* %p1
  load i8, i8* %p1
; CHECK: 3 = MemoryDef(4)
; CHECK-NEXT: store i8 2, i8* %p2
  store i8 2, i8* %p2
; NOLIMIT: MemoryUse(4)
; NOLIMIT-NEXT: load i8, i8* %p1
; LIMIT: MemoryUse(3)
; LIMIT-NEXT: load i8, i8* %p1
  load i8, i8* %p1
  br label %while.cond
}

; CHECK-LABEL: define i32 @use_not_optimized_due_to_backedge
define i32 @use_not_optimized_due_to_backedge(i32* nocapture %m_i_strides, i32* nocapture readonly %eval_left_dims) {
entry:
; CHECK: 1 = MemoryDef(liveOnEntry)
; CHECK-NEXT: store i32 1, i32* %m_i_strides, align 4
  store i32 1, i32* %m_i_strides, align 4
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.inc
  ret i32 %m_i_size.1

for.body:                                         ; preds = %entry, %for.inc
; CHECK: 4 = MemoryPhi({entry,1},{for.inc,3})
; CHECK-NEXT: %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.inc ]
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.inc ]
  %m_i_size.022 = phi i32 [ 1, %entry ], [ %m_i_size.1, %for.inc ]
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %cmp1 = icmp eq i64 %indvars.iv, 0
  %arrayidx2 = getelementptr inbounds i32, i32* %m_i_strides, i64 %indvars.iv
; CHECK: MemoryUse(4)
; CHECK-NEXT: %0 = load i32, i32* %arrayidx2, align 4
  %0 = load i32, i32* %arrayidx2, align 4
  %arrayidx4 = getelementptr inbounds i32, i32* %eval_left_dims, i64 %indvars.iv
; CHECK: MemoryUse(4)
; CHECK-NEXT: %1 = load i32, i32* %arrayidx4, align 4
  %1 = load i32, i32* %arrayidx4, align 4
  %mul = mul nsw i32 %1, %0
  br i1 %cmp1, label %if.then, label %for.inc

if.then:                                          ; preds = %for.body
  %arrayidx7 = getelementptr inbounds i32, i32* %m_i_strides, i64 %indvars.iv.next
; CHECK: 2 = MemoryDef(4)
; CHECK-NEXT: store i32 %mul, i32* %arrayidx7, align 4
  store i32 %mul, i32* %arrayidx7, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body, %if.then
; CHECK: 3 = MemoryPhi({for.body,4},{if.then,2})
; CHECK-NEXT: %m_i_size.1 = phi i32 [ %m_i_size.022, %if.then ], [ %mul, %for.body ]
  %m_i_size.1 = phi i32 [ %m_i_size.022, %if.then ], [ %mul, %for.body ]
  br i1 %cmp1, label %for.body, label %for.cond.cleanup
}


%ArrayType = type { [2 x i64] }
%StructOverArrayType = type { %ArrayType }
%BigStruct = type { i8, i8, i8, i8, i8, i8, i8, %ArrayType, %ArrayType}

; CHECK-LABEL: define void @use_not_optimized_due_to_backedge_unknown
define void @use_not_optimized_due_to_backedge_unknown(%BigStruct* %this) {
entry:
  %eval_left_dims = alloca %StructOverArrayType, align 8
  %tmp0 = bitcast %StructOverArrayType* %eval_left_dims to i8*
  %eval_right_dims = alloca %StructOverArrayType, align 8
  %tmp1 = bitcast %StructOverArrayType* %eval_right_dims to i8*
  %lhs_strides = alloca %ArrayType, align 8
  %rhs_strides = alloca %ArrayType, align 8
  br label %for.body.preheader

for.body.preheader:                               ; preds = %entry
  %arrayidx.i527 = getelementptr inbounds %BigStruct, %BigStruct* %this, i64 0, i32 7, i32 0, i64 0
; CHECK: 1 = MemoryDef(liveOnEntry)
; CHECK-NEXT: store i64 1, i64* %arrayidx.i527, align 8
  store i64 1, i64* %arrayidx.i527, align 8
  %arrayidx.i528 = getelementptr inbounds %BigStruct, %BigStruct* %this, i64 0, i32 8, i32 0, i64 0
; CHECK: 2 = MemoryDef(1)
; CHECK-NEXT: store i64 1, i64* %arrayidx.i528, align 8
  store i64 1, i64* %arrayidx.i528, align 8
  br label %for.main.body

for.main.body:               ; preds = %if.end220.if.then185_crit_edge, %for.body.preheader
; CHECK: 4 = MemoryPhi({for.body.preheader,2},{if.end220.if.then185_crit_edge,3})
; CHECK-NEXT: %nocontract_idx.0656 = phi i64 [ 0, %for.body.preheader ], [ 1, %if.end220.if.then185_crit_edge ]
  %nocontract_idx.0656 = phi i64 [ 0, %for.body.preheader ], [ 1, %if.end220.if.then185_crit_edge ]
  %add199 = add nuw nsw i64 %nocontract_idx.0656, 1
  %cmp200 = icmp eq i64 %nocontract_idx.0656, 0
  %arrayidx.i559 = getelementptr inbounds %BigStruct, %BigStruct* %this, i64 0, i32 7, i32 0, i64 %nocontract_idx.0656
; CHECK: MemoryUse(4)
; CHECK-NEXT: %tmp21 = load i64, i64* %arrayidx.i559, align 8
  %tmp21 = load i64, i64* %arrayidx.i559, align 8
  %mul206 = mul nsw i64 %tmp21, %tmp21
  br i1 %cmp200, label %if.end220.if.then185_crit_edge, label %the.end

if.end220.if.then185_crit_edge:                   ; preds = %for.main.body
  %arrayidx.i571 = getelementptr inbounds %BigStruct, %BigStruct* %this, i64 0, i32 7, i32 0, i64 %add199
; CHECK: 3 = MemoryDef(4)
; CHECK-NEXT: store i64 %mul206, i64* %arrayidx.i571, align 8
  store i64 %mul206, i64* %arrayidx.i571, align 8
  br label %for.main.body

the.end:                            ; preds = %for.main.body
  ret void

}


@c = local_unnamed_addr global [2 x i16] zeroinitializer, align 2

define i32 @dont_merge_noalias_simple(i32* noalias %ptr) {
; CHECK-LABEL: define i32 @dont_merge_noalias_simple
; CHECK-LABEL: entry:
; CHECK:       ; 1 = MemoryDef(liveOnEntry)
; CHECK-NEXT:  store i16 1, i16* %s1.ptr, align 2

; CHECK-LABEL: %for.body
; CHECK:       ; MemoryUse(4)
; CHECK-NEXT:    %lv = load i16, i16* %arrayidx, align 2

entry:
  %s1.ptr = getelementptr inbounds [2 x i16], [2 x i16]* @c, i64 0, i64 0
  store i16 1, i16* %s1.ptr, align 2
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %storemerge2 = phi i32 [ 1, %entry ], [ %dec, %for.body ]
  %idxprom1 = zext i32 %storemerge2 to i64
  %arrayidx = getelementptr inbounds [2 x i16], [2 x i16]* @c, i64 0, i64 %idxprom1
  %lv = load i16, i16* %arrayidx, align 2
  %conv = sext i16 %lv to i32
  store i32 %conv, i32* %ptr, align 4
  %dec = add nsw i32 %storemerge2, -1
  %cmp = icmp sgt i32 %storemerge2, 0
  br i1 %cmp, label %for.body, label %for.end

for.end:                                          ; preds = %for.body
  %s2.ptr = getelementptr inbounds [2 x i16], [2 x i16]* @c, i64 0, i64 0
  store i16 0, i16* %s2.ptr, align 2
  ret i32 0
}


define i32 @dont_merge_noalias_complex(i32* noalias %ptr, i32* noalias %another) {
; CHECK-LABEL: define i32 @dont_merge_noalias_complex
; CHECK-LABEL: entry:
; CHECK:       ; 1 = MemoryDef(liveOnEntry)
; CHECK-NEXT:  store i16 1, i16* %s1.ptr, align 2

; CHECK-LABEL: %for.body
; CHECK:       ; MemoryUse(7)
; CHECK-NEXT:    %lv = load i16, i16* %arrayidx, align 2

entry:
  %s1.ptr = getelementptr inbounds [2 x i16], [2 x i16]* @c, i64 0, i64 0
  store i16 1, i16* %s1.ptr, align 2
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %storemerge2 = phi i32 [ 1, %entry ], [ %dec, %merge.body ]
  %idxprom1 = zext i32 %storemerge2 to i64
  %arrayidx = getelementptr inbounds [2 x i16], [2 x i16]* @c, i64 0, i64 %idxprom1
  %lv = load i16, i16* %arrayidx, align 2
  %conv = sext i16 %lv to i32
  store i32 %conv, i32* %ptr, align 4
  %dec = add nsw i32 %storemerge2, -1

  %cmpif = icmp sgt i32 %storemerge2, 1
  br i1 %cmpif, label %if.body, label %else.body

if.body:
  store i32 %conv, i32* %another, align 4
  br label %merge.body

else.body:
  store i32 %conv, i32* %another, align 4
  br label %merge.body

merge.body:
  %cmp = icmp sgt i32 %storemerge2, 0
  br i1 %cmp, label %for.body, label %for.end

for.end:                                          ; preds = %for.body
  %s2.ptr = getelementptr inbounds [2 x i16], [2 x i16]* @c, i64 0, i64 0
  store i16 0, i16* %s2.ptr, align 2
  ret i32 0
}

declare i1 @should_exit(i32) readnone
declare void @init([32 x i32]*)

; Test case for PR47498.
; %l.1 may read the result of `store i32 10, i32* %p.1` in %storebb, because
; after %storebb has been executed, %loop.1.header might be executed again.
; Make sure %l.1's defining access is the MemoryPhi in the block.
define void @dont_merge_noalias_complex_2(i32 %arg, i32 %arg1)  {
; CHECK-LABEL: define void @dont_merge_noalias_complex_2(

; CHECK-LABEL: entry:
; CHECK:       ; 1 = MemoryDef(liveOnEntry)
; CHECK-NEXT:  call void @init([32 x i32]* %tmp)

; CHECK-LABEL: loop.1.header:
; CHECK-NEXT:  ; 4 = MemoryPhi({entry,1},{loop.1.latch,3})
; CHECK:       ; MemoryUse(4)
; CHECK-NEXT:  %l.1 = load i32, i32* %p.1, align 4

; CHECK-LABEL: loop.1.latch:
; CHECK-NEXT:  ; 3 = MemoryPhi({loop.1.header,4},{storebb,2})

; CHECK-LABEL: storebb:
; CHECK-NEXT:  %iv.add2 = add nuw nsw i64 %iv, 2
; CHECK-NEXT:  %p.2 = getelementptr inbounds [32 x i32], [32 x i32]* %tmp, i64 0, i64 %iv.add2
; CHECK-NEXT:  ; MemoryUse(4)
; CHECK-NEXT:  %l.2 = load i32, i32* %p.2, align 4
; CHECK-NEXT:  ; 2 = MemoryDef(4)
; CHECK-NEXT:  store i32 10, i32* %p.1, align 4
entry:
  %tmp = alloca [32 x i32], align 16
  call void @init([32 x i32]* %tmp)
  br label %loop.1.header

loop.1.header:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop.1.latch ]
  %iv.next = add nuw nsw i64 %iv, 1
  %p.1 = getelementptr inbounds [32 x i32], [32 x i32]* %tmp, i64 0, i64 %iv.next
  %l.1 = load i32, i32* %p.1, align 4
  %tmp244 = icmp ult i64 %iv, 10
  br i1 %tmp244, label %loop.1.latch, label %storebb

loop.1.latch:
  %ec = call i1 @should_exit(i32 %l.1)
  br i1 %ec, label %exit, label %loop.1.header

storebb:
  %iv.add2 = add nuw nsw i64 %iv, 2
  %p.2 = getelementptr inbounds [32 x i32], [32 x i32]* %tmp, i64 0, i64 %iv.add2
  %l.2 = load i32, i32* %p.2, align 4
  store i32 10, i32* %p.1, align 4
  br label %loop.1.latch

exit:
  ret void
}

define i32 @phi_with_constant_values(i1 %cmp) {
; CHECK-LABEL: define i32 @phi_with_constant_values
; CHECK-LABEL: lhs:
; CHECK:       ; 1 = MemoryDef(liveOnEntry)
; CHECK-NEXT:  store i16 1, i16* %s1.ptr, align 2

; CHECK-LABEL: rhs:
; CHECK:       ; 2 = MemoryDef(liveOnEntry)
; CHECK-NEXT:  store i16 1, i16* %s2.ptr, align 2

; CHECK-LABEL: merge:
; CHECK:       ; 3 = MemoryPhi({lhs,1},{rhs,2})
; CHECK-NEXT   %storemerge2 = phi i32 [ 2, %lhs ], [ 3, %rhs ]
; TODO: Here the defining access could be liveOnEntry. Since all incoming
; values of phi node can be phi translated.
; CHECK:       ; MemoryUse(3)
; CHECK-NEXT:    %lv = load i16, i16* %arrayidx, align 2

entry:
  br i1 %cmp, label %lhs, label %rhs

lhs:
  %s1.ptr = getelementptr inbounds [2 x i16], [2 x i16]* @c, i64 0, i64 0
  store i16 1, i16* %s1.ptr, align 2
  br label %merge

rhs:
  %s2.ptr = getelementptr inbounds [2 x i16], [2 x i16]* @c, i64 0, i64 1
  store i16 1, i16* %s2.ptr, align 2
  br label %merge

merge:                                         ; preds = %for.body, %entry
  %storemerge2 = phi i32 [ 2, %lhs ], [ 3, %rhs ]
  %idxprom1 = zext i32 %storemerge2 to i64
  %arrayidx = getelementptr inbounds [2 x i16], [2 x i16]* @c, i64 0, i64 %idxprom1
  %lv = load i16, i16* %arrayidx, align 2
  br label %end

end:                                          ; preds = %for.body
  ret i32 0
}

; CHECK-LABEL: define void @use_clobbered_by_def_in_loop()
define void @use_clobbered_by_def_in_loop() {
entry:
  %nodeStack = alloca [12 x i32], align 4
  %0 = bitcast [12 x i32]* %nodeStack to i8*
  call void @llvm.lifetime.start.p0i8(i64 48, i8* nonnull %0)
  br i1 false, label %cleanup, label %while.cond

; CHECK-LABEL: while.cond:
; CHECK-NEXT: ; [[NO6:.*]] = MemoryPhi({entry,1},{while.cond.backedge,5})

while.cond:                                       ; preds = %entry, %while.cond.backedge
  %depth.1 = phi i32 [ %depth.1.be, %while.cond.backedge ], [ 0, %entry ]
  %cmp = icmp sgt i32 %depth.1, 0
  br i1 %cmp, label %land.rhs, label %while.end

; CHECK-LABEL: land.rhs:
; CHECK-NEXT: %sub = add nsw i32 %depth.1, -1
; CHECK-NEXT: %arrayidx = getelementptr inbounds [12 x i32], [12 x i32]* %nodeStack, i32 0, i32 %sub
; CHECK-NEXT: ; MemoryUse([[NO6]])
; CHECK-NEXT: %1 = load i32, i32* %arrayidx, align 4

land.rhs:                                         ; preds = %while.cond
  %sub = add nsw i32 %depth.1, -1
  %arrayidx = getelementptr inbounds [12 x i32], [12 x i32]* %nodeStack, i32 0, i32 %sub
  %1 = load i32, i32* %arrayidx, align 4
  br i1 true, label %while.body, label %while.end

while.body:                                       ; preds = %land.rhs
  br i1 true, label %cleanup, label %while.cond.backedge

while.cond.backedge:                              ; preds = %while.body, %while.end
  %depth.1.be = phi i32 [ %sub, %while.body ], [ %inc, %while.end ]
  br label %while.cond

while.end:                                        ; preds = %while.cond, %land.rhs
  %arrayidx10 = getelementptr inbounds [12 x i32], [12 x i32]* %nodeStack, i32 0, i32 %depth.1
  store i32 %depth.1, i32* %arrayidx10, align 4
  %inc = add nsw i32 %depth.1, 1
  br i1 true, label %cleanup, label %while.cond.backedge

cleanup:                                          ; preds = %while.body, %while.end, %entry
  call void @llvm.lifetime.end.p0i8(i64 48, i8* nonnull %0)
  ret void
}

declare void @llvm.lifetime.start.p0i8(i64 immarg, i8* nocapture)
declare void @llvm.lifetime.end.p0i8(i64 immarg, i8* nocapture)

define void @another_loop_clobber_inc() {
; CHECK-LABEL: void @another_loop_clobber_inc
; CHECK-LABEL: loop.header:
; CHECK-NEXT:  ; 4 = MemoryPhi({entry,1},{cond.read,3})

; CHECK-LABEL: cond.read:
; CHECK:       ; MemoryUse(4)
; CHECK-NEXT:  %use = load i32, i32* %ptr.1, align 4
; CHECK-NEXT:  ; 2 = MemoryDef(4)
; CHECK-NEXT:  %c.2 = call i1 @cond(i32 %use)
; CHECK-NEXT:  %ptr.10 = getelementptr inbounds [12 x i32], [12 x i32]* %nodeStack, i32 0, i32 %inc
; CHECK-NEXT:  ; 3 = MemoryDef(2)
; CHECK-NEXT:  store i32 10, i32* %ptr.2, align 4

entry:
  %nodeStack = alloca [12 x i32], align 4
  %c.1 = call i1 @cond(i32 1)
  br i1 %c.1, label %cleanup, label %loop.header

loop.header:                                       ; preds = %entry, %while.cond.backedge
  %depth.1 = phi i32 [ %inc, %cond.read], [ 1, %entry ]
  %cmp = icmp sgt i32 %depth.1, 0
  %inc = add nsw i32 %depth.1, 3
  %inc2 = add nsw i32 %depth.1, 6
  br i1 %cmp, label %cond.read, label %cleanup

cond.read:                                        ; preds = %while.cond
  %ptr.1 = getelementptr inbounds [12 x i32], [12 x i32]* %nodeStack, i32 0, i32 %depth.1
  %ptr.2 = getelementptr inbounds [12 x i32], [12 x i32]* %nodeStack, i32 0, i32 %inc2
  %use = load i32, i32* %ptr.1, align 4
  %c.2 = call i1 @cond(i32 %use)
  %ptr.10 = getelementptr inbounds [12 x i32], [12 x i32]* %nodeStack, i32 0, i32 %inc
  store i32 10, i32* %ptr.2, align 4
  br i1 %c.2, label %loop.header, label %cleanup

cleanup:
  ret void
}

define void @another_loop_clobber_dec() {
; CHECK-LABEL: void @another_loop_clobber_dec
; CHECK-LABEL: loop.header:
; CHECK-NEXT:  ; 4 = MemoryPhi({entry,1},{cond.read,3})

; CHECK-LABEL: cond.read:
; CHECK:       ; MemoryUse(4)
; CHECK-NEXT:  %use = load i32, i32* %ptr.1, align 4
; CHECK-NEXT:  ; 2 = MemoryDef(4)
; CHECK-NEXT:  %c.2 = call i1 @cond(i32 %use)
; CHECK-NEXT:  %ptr.10 = getelementptr inbounds [12 x i32], [12 x i32]* %nodeStack, i32 0, i64 %sub
; CHECK-NEXT:  ; 3 = MemoryDef(2)
; CHECK-NEXT:  store i32 10, i32* %ptr.2, align 4

entry:
  %nodeStack = alloca [12 x i32], align 4
  %c.1 = call i1 @cond(i32 1)
  br i1 %c.1, label %cleanup, label %loop.header

loop.header:                                       ; preds = %entry, %while.cond.backedge
  %depth.1 = phi i64 [ %sub, %cond.read], [ 20, %entry ]
  %cmp = icmp sgt i64 %depth.1, 6
  %sub = sub nsw nuw i64 %depth.1, 3
  %sub2 = sub nsw nuw i64 %depth.1, 6
  br i1 %cmp, label %cond.read, label %cleanup

cond.read:                                        ; preds = %while.cond
  %ptr.1 = getelementptr inbounds [12 x i32], [12 x i32]* %nodeStack, i32 0, i64 %depth.1
  %ptr.2 = getelementptr inbounds [12 x i32], [12 x i32]* %nodeStack, i32 0, i64 %sub2
  %use = load i32, i32* %ptr.1, align 4
  %c.2 = call i1 @cond(i32 %use)
  %ptr.10 = getelementptr inbounds [12 x i32], [12 x i32]* %nodeStack, i32 0, i64 %sub
  store i32 10, i32* %ptr.2, align 4
  br i1 %c.2, label %loop.header, label %cleanup

cleanup:
  ret void
}

declare i1 @cond(i32)
