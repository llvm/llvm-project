; RUN: opt -aa-pipeline=basic-aa -passes='print<memoryssa>' -verify-memoryssa -disable-output -memssa-check-limit=0 < %s 2>&1 | FileCheck %s
target triple = "amdgpu6.00"

@g4 = external unnamed_addr constant i8, align 1

define signext i8 @cmp_constant(ptr %q, i8 %v) local_unnamed_addr {
entry:

  store i8 %v, ptr %q, align 1
; CHECK: 1 = MemoryDef(liveOnEntry)
; CHECK-NEXT: store i8 %v, ptr %q, align 1

  %0 = load i8, ptr @g4, align 1
; Make sure that this load is liveOnEntry just based on the fact that @g4 is
; constant memory.
; CHECK: MemoryUse(liveOnEntry)
; CHECK-NEXT: load i8, ptr @g4, align 1

  ret i8 %0
}

