; RUN: opt -basicaa -print-memoryssa -verify-memoryssa -analyze < %s 2>&1 | FileCheck %s
; RUN: opt -aa-pipeline=basic-aa -passes='print<memoryssa>,verify<memoryssa>' -disable-output < %s 2>&1 | FileCheck %s
;
; This test ensures we don't end up with multiple reaching defs for a single
; use/phi edge If we were to optimize defs, we would end up with 2=
; MemoryDef(liveOnEntry) and 4 = MemoryDef(liveOnEntry) Both would mean both
; 1,2, and 3,4 would reach the phi node.  Because the phi node can only have one
; entry on each edge, it would choose 2, 4 and disconnect 1 and 3 completely
; from the SSA graph, even though they are not dead

define void @sink_store(i32 %index, i32* %foo, i32* %bar) {
entry:
  %cmp = trunc i32 %index to i1
  br i1 %cmp, label %if.then, label %if.else

if.then:                                          ; preds = %entry
; CHECK: 1 = MemoryDef(liveOnEntry)
; CHECK-NEXT:   store i32 %index, i32* %foo, align 4
  store i32 %index, i32* %foo, align 4
; CHECK: 2 = MemoryDef(1)
; CHECK-NEXT:   store i32 %index, i32* %bar, align 4
  store i32 %index, i32* %bar, align 4
  br label %if.end

if.else:                                          ; preds = %entry
; CHECK: 3 = MemoryDef(liveOnEntry)
; CHECK-NEXT:   store i32 %index, i32* %foo, align 4
  store i32 %index, i32* %foo, align 4
; CHECK: 4 = MemoryDef(3)
; CHECK-NEXT:   store i32 %index, i32* %bar, align 4
  store i32 %index, i32* %bar, align 4
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
; CHECK: 5 = MemoryPhi({if.then,2},{if.else,4})
; CHECK: MemoryUse(5)
; CHECK-NEXT:   %c = load i32, i32* %foo
  %c = load i32, i32* %foo
; CHECK: MemoryUse(5)
; CHECK-NEXT:   %d = load i32, i32* %bar
  %d = load i32, i32* %bar
  ret void
}
