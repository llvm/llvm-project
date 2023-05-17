; RUN: opt -aa-pipeline=basic-aa -passes='print<memoryssa>,verify<memoryssa>' -disable-output < %s 2>&1 | FileCheck %s
;
; Ensures that assumes are treated as not reading or writing memory.

declare void @llvm.assume(i1)

define i32 @foo(ptr %a, ptr %b, i1 %c) {
; CHECK: 1 = MemoryDef(liveOnEntry)
; CHECK-NEXT: store i32 4
  store i32 4, ptr %a, align 4
; CHECK-NOT: MemoryDef
; CHECK: call void @llvm.assume
  call void @llvm.assume(i1 %c)
; CHECK: MemoryUse(1)
; CHECK-NEXT: %1 = load i32
  %1 = load i32, ptr %a, align 4
  ret i32 %1
}
