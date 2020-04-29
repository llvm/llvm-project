; RUN: opt -basicaa -print-memoryssa -verify-memoryssa -analyze < %s 2>&1 | FileCheck %s --check-prefixes=CHECK,NOLIMIT
; RUN: opt -memssa-check-limit=0 -basicaa -print-memoryssa -verify-memoryssa -analyze < %s 2>&1 | FileCheck %s --check-prefixes=CHECK,LIMIT
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
