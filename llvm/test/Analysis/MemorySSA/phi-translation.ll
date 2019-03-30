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

