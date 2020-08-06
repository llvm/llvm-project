; RUN: opt -basic-aa -print-memoryssa -verify-memoryssa -enable-new-pm=0 -analyze < %s 2>&1 | FileCheck %s --check-prefixes=CHECK,NOLIMIT
; RUN: opt -memssa-check-limit=0 -basic-aa -print-memoryssa -verify-memoryssa -enable-new-pm=0 -analyze < %s 2>&1 | FileCheck %s --check-prefixes=CHECK,LIMIT
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
; NOLIMIT: MemoryUse(1) MayAlias
; NOLIMIT-NEXT: load i8, i8* %local, align 1
; LIMIT: MemoryUse(3) MayAlias
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
; NOLIMIT: MemoryUse(1) MayAlias
; NOLIMIT-NEXT: load i8, i8* %local
; LIMIT: MemoryUse(6) MayAlias
; LIMIT-NEXT: load i8, i8* %local
  load i8, i8* %local
  ret void
}

; CHECK-LABEL: define void @cross_phi
define void @cross_phi(i8* noalias %p1, i8* noalias %p2) {
; CHECK: 1 = MemoryDef(liveOnEntry)
; CHECK-NEXT: store i8 0, i8* %p1
  store i8 0, i8* %p1
; NOLIMIT: MemoryUse(1) MustAlias
; NOLIMIT-NEXT: load i8, i8* %p1
; LIMIT: MemoryUse(1) MayAlias
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
; NOLIMIT: MemoryUse(1) MustAlias
; NOLIMIT-NEXT: load i8, i8* %p1
; LIMIT: MemoryUse(8) MayAlias
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
; NOLIMIT: MemoryUse(1) MayAlias
; NOLIMIT-NEXT: load i8, i8* %p1
; LIMIT: MemoryUse(4) MayAlias
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
; CHECK: MemoryUse(4) MayAlias
; CHECK-NEXT: load i8, i8* %p1
  load i8, i8* %p1
; CHECK: 3 = MemoryDef(4)
; CHECK-NEXT: store i8 2, i8* %p2
  store i8 2, i8* %p2
; NOLIMIT: MemoryUse(4) MayAlias
; NOLIMIT-NEXT: load i8, i8* %p1
; LIMIT: MemoryUse(3) MayAlias
; LIMIT-NEXT: load i8, i8* %p1
  load i8, i8* %p1
  br label %while.cond
}

; CHECK-LABEL: define i32 @use_not_optimized_due_to_backedge
define i32 @use_not_optimized_due_to_backedge(i32* nocapture %m_i_strides, i32* nocapture readonly %eval_left_dims) {
entry:
; CHECK: 1 = MemoryDef(liveOnEntry)
; CHECK_NEXT: store i32 1, i32* %m_i_strides, align 4
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
; CHECK: MemoryUse(4) MayAlias
; CHECK-NEXT: %0 = load i32, i32* %arrayidx2, align 4
  %0 = load i32, i32* %arrayidx2, align 4
  %arrayidx4 = getelementptr inbounds i32, i32* %eval_left_dims, i64 %indvars.iv
; CHECK: MemoryUse(4) MayAlias
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
; CHECK: MemoryUse(4) MayAlias
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
; CHECK:       ; MemoryUse(4) MayAlias
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
; CHECK:       ; MemoryUse(7) MayAlias
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

