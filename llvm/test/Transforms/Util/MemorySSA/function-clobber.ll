; RUN: opt -basicaa -memoryssa -analyze < %s 2>&1 | FileCheck %s
; RUN: opt -aa-pipeline=basic-aa -passes='print<memoryssa>,verify<memoryssa>' -disable-output < %s 2>&1 | FileCheck %s
;
; Ensuring that external functions without attributes are MemoryDefs

@g = external global i32
declare void @modifyG()

define i32 @foo() {
; CHECK: MemoryUse(liveOnEntry)
; CHECK-NEXT: %1 = load i32
  %1 = load i32, i32* @g

; CHECK: 1 = MemoryDef(liveOnEntry)
; CHECK-NEXT: store i32 4
  store i32 4, i32* @g, align 4

; CHECK: 2 = MemoryDef(1)
; CHECK-NEXT: call void @modifyG()
  call void @modifyG()

; CHECK: MemoryUse(2)
; CHECK-NEXT: %2 = load i32
  %2 = load i32, i32* @g
  %3 = add i32 %2, %1
  ret i32 %3
}

declare void @readEverything() readonly
declare void @clobberEverything()

; CHECK-LABEL: define void @bar
define void @bar() {
; CHECK: 1 = MemoryDef(liveOnEntry)
; CHECK-NEXT: call void @clobberEverything()
  call void @clobberEverything()
  br i1 undef, label %if.end, label %if.then

if.then:
; CHECK: MemoryUse(1)
; CHECK-NEXT: call void @readEverything()
  call void @readEverything()
; CHECK: 2 = MemoryDef(1)
; CHECK-NEXT: call void @clobberEverything()
  call void @clobberEverything()
  br label %if.end

if.end:
; CHECK: 3 = MemoryPhi({%0,1},{if.then,2})
; CHECK: MemoryUse(3)
; CHECK-NEXT: call void @readEverything()
  call void @readEverything()
  ret void
}
