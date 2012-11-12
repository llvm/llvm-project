; RUN: llc < %s -march=arm -mcpu=cortex-a8 | FileCheck %s
; Check that memcpy gets lowered to ldm/stm, at least in this very smple case.

%struct.Foo = type { i32, i32, i32, i32 }

define void @_Z10CopyStructP3FooS0_(%struct.Foo* nocapture %a, %struct.Foo* nocapture %b) nounwind {
entry:
;CHECK: ldm
;CHECK: stm
  %0 = bitcast %struct.Foo* %a to i8*
  %1 = bitcast %struct.Foo* %b to i8*
  tail call void @llvm.memcpy.p0i8.p0i8.i32(i8* %0, i8* %1, i32 16, i32 4, i1 false)
  ret void
}

declare void @llvm.memcpy.p0i8.p0i8.i32(i8* nocapture, i8* nocapture, i32, i32, i1) nounwind
