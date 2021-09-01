; RUN: opt -aa-pipeline=basic-aa -passes='print<memoryssa>,verify<memoryssa>' -disable-output < %s 2>&1 | FileCheck %s
;
; Checks that basicAA is doing some amount of disambiguation for us

define i32 @foo(i1 %cond) {
  %a = alloca i32, align 4
  %b = alloca i32, align 4
; CHECK: 1 = MemoryDef(liveOnEntry)
; CHECK-NEXT: store i32 0
  store i32 0, i32* %a, align 4
; CHECK: 2 = MemoryDef(1)
; CHECK-NEXT: store i32 1
  store i32 1, i32* %b, align 4

; CHECK: MemoryUse(1)
; CHECK-NEXT: %1 = load i32
  %1 = load i32, i32* %a, align 4
; CHECK: MemoryUse(2)
; CHECK-NEXT: %2 = load i32
  %2 = load i32, i32* %b, align 4

  %3 = add i32 %1, %2
  ret i32 %3
}
