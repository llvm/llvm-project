; RUN: opt -passes='loop-mssa(simple-loop-unswitch<nontrivial>),print<memoryssa>' -verify-memoryssa -disable-output -S < %s 2>&1 | FileCheck %s

; CHECK: preds = %bb2{{$}}
; CHECK-NEXT: MemoryDef
; CHECK-NEXT: call i32 @bar()

define i32 @foo(i1 %arg, ptr %arg1) {
bb:
  br label %bb2

bb2:                                              ; preds = %bb2, %bb
  %i = select i1 %arg, ptr %arg1, ptr @bar
  %i3 = call i32 %i()
  br i1 %arg, label %bb2, label %bb4

bb4:                                              ; preds = %bb2
  ret i32 %i3
}

declare i32 @bar() nounwind willreturn memory(none)
