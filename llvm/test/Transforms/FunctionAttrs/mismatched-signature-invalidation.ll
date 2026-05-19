; RUN: opt -disable-output -passes="function(print<memoryssa>),cgscc(function-attrs),function(print<memoryssa>)" < %s 2>&1 | FileCheck %s

@g = external global i16

define i16 @fn() {
  %v = load i16, ptr @g
  ret i16 %v
}

declare void @fn2(i16)

; CHECK-LABEL: MemorySSA for function: test
; CHECK: 1 = MemoryDef(3)
; CHECK-NEXT: %call = call i16 @fn(i32 0)

; CHECK-LABEL: MemorySSA for function: test
; CHECK: MemoryUse(2)
; CHECK-NEXT: %call = call i16 @fn(i32 0)

define void @test() {
entry:
  br label %loop

loop:
  %call = call i16 @fn(i32 0) ; intentional signature mismatch
  call void @fn2(i16 %call)
  br i1 false, label %loop, label %exit

exit:
  ret void
}
