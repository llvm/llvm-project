; RUN: opt -passes='print<memoryssa>' -disable-output < %s 2>&1 | FileCheck %s

; FIXME: The %sel2 load should be MemoryUse(4), because it loads the value
; stored by the %sel1 store on the previous iteration.
; CHECK: 1 = MemoryDef(liveOnEntry)
; CHECK-NEXT: store i32 1, ptr %a1, align 4
; CHECK: 2 = MemoryDef(1)
; CHECK-NEXT: store i32 2, ptr %a2, align 4
; CHECK: 4 = MemoryPhi({entry,2},{loop,3})
; CHECK-NEXT: %c = phi i1 [ true, %entry ], [ false, %loop ]
; CHECK: 3 = MemoryDef(4)
; CHECK-NEXT: store i32 0, ptr %sel1, align 4
; CHECK: MemoryUse(2)
; CHECK-NEXT: %v = load i32, ptr %sel2, align 4
define void @test() {
entry:
  %a1 = alloca i32
  %a2 = alloca i32
  store i32 1, ptr %a1
  store i32 2, ptr %a2
  br label %loop

loop:
  %c = phi i1 [ true, %entry ], [ false, %loop ]
  %sel1 = select i1 %c, ptr %a1, ptr %a2
  store i32 0, ptr %sel1
  %sel2 = select i1 %c, ptr %a2, ptr %a1
  %v = load i32, ptr %sel2
  br label %loop
}
