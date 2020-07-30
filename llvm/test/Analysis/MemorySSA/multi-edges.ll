; RUN: opt -basic-aa -print-memoryssa -verify-memoryssa -analyze < %s 2>&1 | FileCheck %s
; RUN: opt -aa-pipeline=basic-aa -passes='print<memoryssa>,verify<memoryssa>' -disable-output < %s 2>&1 | FileCheck %s
;
; Makes sure we have a sane model if both successors of some block is the same
; block.

define i32 @foo(i1 %a) {
entry:
  %0 = alloca i32, align 4
; CHECK: 1 = MemoryDef(liveOnEntry)
; CHECK-NEXT: store i32 4
  store i32 4, i32* %0
  br i1 %a, label %Loop.Body, label %Loop.End

Loop.Body:
; CHECK: 4 = MemoryPhi({entry,1},{Loop.End,3})
; CHECK-NEXT: 2 = MemoryDef(4)
; CHECK-NEXT: store i32 5
  store i32 5, i32* %0, align 4
  br i1 %a, label %Loop.End, label %Loop.End ; WhyDoWeEvenHaveThatLever.gif

Loop.End:
; CHECK: 3 = MemoryPhi({entry,1},{Loop.Body,2},{Loop.Body,2})
; CHECK-NEXT: MemoryUse(3)
; CHECK-NEXT: %1 = load
  %1 = load i32, i32* %0, align 4
  %2 = icmp eq i32 5, %1
  br i1 %2, label %Ret, label %Loop.Body

Ret:
  ret i32 %1
}
