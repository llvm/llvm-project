; RUN: opt -aa-pipeline=basic-aa -passes='print<memoryssa>' -verify-memoryssa < %s 2>&1 | FileCheck %s
;
; Things that BasicAA can prove points to constant memory should be
; liveOnEntry, as well.

declare void @clobberAllTheThings()

@str = private unnamed_addr constant [2 x i8] c"hi"

define i8 @foo() {
; CHECK: 1 = MemoryDef(liveOnEntry)
; CHECK-NEXT: call void @clobberAllTheThings()
  call void @clobberAllTheThings()
; CHECK: MemoryUse(liveOnEntry)
; CHECK-NEXT: %1 = load i8
  %1 = load i8, ptr @str, align 1
  %2 = getelementptr [2 x i8], ptr @str, i64 0, i64 1
; CHECK: MemoryUse(liveOnEntry)
; CHECK-NEXT: %3 = load i8
  %3 = load i8, ptr %2, align 1
  %4 = add i8 %1, %3
  ret i8 %4
}

define i8 @select(i1 %b) {
  %1 = alloca i8, align 1
; CHECK: 1 = MemoryDef(liveOnEntry)
; CHECK-NEXT: store i8 0
  store i8 0, ptr %1, align 1

; CHECK: 2 = MemoryDef(1)
; CHECK-NEXT: call void @clobberAllTheThings()
  call void @clobberAllTheThings()
  %2 = select i1 %b, ptr @str, ptr %1
; CHECK: MemoryUse(2)
; CHECK-NEXT: %3 = load i8
  %3 = load i8, ptr %2, align 1
  ret i8 %3
}
