; RUN: opt -basicaa -print-memoryssa -verify-memoryssa -analyze < %s 2>&1 | FileCheck %s
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
