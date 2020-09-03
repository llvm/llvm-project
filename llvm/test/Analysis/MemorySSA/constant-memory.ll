; RUN: opt -basic-aa -print-memoryssa -verify-memoryssa -analyze < %s 2>&1 | FileCheck %s
;
; Things that BasicAA can prove points to constant memory should be
; liveOnEntry, as well.

declare void @clobberAllTheThings()

@str = private unnamed_addr constant [2 x i8] c"hi"

define i8 @foo() {
; CHECK: 1 = MemoryDef(liveOnEntry)
; CHECK-NEXT: call void @clobberAllTheThings()
  call void @clobberAllTheThings()
  %1 = getelementptr [2 x i8], [2 x i8]* @str, i64 0, i64 0
; CHECK: MemoryUse(liveOnEntry)
; CHECK-NEXT: %2 = load i8
  %2 = load i8, i8* %1, align 1
  %3 = getelementptr [2 x i8], [2 x i8]* @str, i64 0, i64 1
; CHECK: MemoryUse(liveOnEntry)
; CHECK-NEXT: %4 = load i8
  %4 = load i8, i8* %3, align 1
  %5 = add i8 %2, %4
  ret i8 %5
}

define i8 @select(i1 %b) {
  %1 = alloca i8, align 1
; CHECK: 1 = MemoryDef(liveOnEntry)
; CHECK-NEXT: store i8 0
  store i8 0, i8* %1, align 1

; CHECK: 2 = MemoryDef(1)
; CHECK-NEXT: call void @clobberAllTheThings()
  call void @clobberAllTheThings()
  %2 = getelementptr [2 x i8], [2 x i8]* @str, i64 0, i64 0
  %3 = select i1 %b, i8* %2, i8* %1
; CHECK: MemoryUse(2)
; CHECK-NEXT: %4 = load i8
  %4 = load i8, i8* %3, align 1
  ret i8 %4
}
