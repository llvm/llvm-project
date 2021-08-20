; RUN: not --crash opt -passes='loop-mssa(loop-unroll-full)' 2>&1 < %s | FileCheck %s

; CHECK: LLVM ERROR: Loop pass manager using MemorySSA contains a pass that does not preserve MemorySSA

define void @test() {
entry:
  br label %loop

loop:
  %i = phi i32 [ 0, %entry ], [ %i.inc, %loop ]
  %i.inc = add i32 %i, 1
  %c = icmp ult i32 %i, 8
  br i1 %c, label %loop, label %exit

exit:
  ret void
}
