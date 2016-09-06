; RUN: opt -basicaa -print-memoryssa -verify-memoryssa -analyze < %s 2>&1 | FileCheck %s
; RUN: opt -aa-pipeline=basic-aa -passes='print<memoryssa>,verify<memoryssa>' -disable-output < %s 2>&1 | FileCheck %s
define void @F(i8*) {
  br i1 true, label %left, label %right
left:
; CHECK: 1 = MemoryDef(liveOnEntry)
  store i8 16, i8* %0
  br label %merge
right:
  br label %merge

merge:
; CHECK-NOT: 2 = MemoryPhi
ret void 
}

define void @F2(i8*) {
  br i1 true, label %left, label %right
left:
; CHECK: 1 = MemoryDef(liveOnEntry)
  store i8 16, i8* %0
  br label %merge
right:
  br label %merge

merge:
; CHECK: 2 = MemoryPhi({left,1},{right,liveOnEntry})
%c = load i8, i8* %0
ret void 
}

; Ensure we treat def-only blocks as though they have uses for phi placement.
; CHECK-LABEL: define void @F3
define void @F3() {
  %a = alloca i8
; CHECK: 1 = MemoryDef(liveOnEntry)
; CHECK-NEXT: store i8 0, i8* %a
  store i8 0, i8* %a
  br i1 undef, label %if.then, label %if.end

if.then:
; CHECK: 2 = MemoryDef(1)
; CHECK-NEXT: store i8 1, i8* %a
  store i8 1, i8* %a
  br label %if.end

if.end:
; CHECK: 4 = MemoryPhi({%0,1},{if.then,2})
; CHECK: 3 = MemoryDef(4)
; CHECK-NEXT: store i8 2, i8* %a
  store i8 2, i8* %a
  ret void
}
