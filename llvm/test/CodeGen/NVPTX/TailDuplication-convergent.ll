; RUN: llc -O2 -tail-dup-size=100 -enable-tail-merge=0 < %s | FileCheck %s
; RUN: %if ptxas %{ llc -O2 -tail-dup-size=100 -enable-tail-merge=0 < %s | %ptxas-verify %}
target triple = "nvptx64-nvidia-cuda"

declare void @foo()
declare void @llvm.nvvm.barrier0()

; syncthreads shouldn't be duplicated.
; CHECK: .func call_syncthreads
; CHECK: bar.sync
; CHECK-NOT: bar.sync
define void @call_syncthreads(i32* %a, i32* %b, i1 %cond, i1 %cond2) nounwind {
  br i1 %cond, label %L1, label %L2
  br i1 %cond2, label %Ret, label %L1
Ret:
  ret void
L1:
  store i32 0, i32* %a
  br label %L42
L2:
  store i32 1, i32* %a
  br label %L42
L42:
  call void @llvm.nvvm.barrier0()
  br label %Ret
}

; Check that call_syncthreads really does trigger tail duplication.
; CHECK: .func call_foo
; CHECK: call
; CHECK: call
define void @call_foo(i32* %a, i32* %b, i1 %cond, i1 %cond2) nounwind {
  br i1 %cond, label %L1, label %L2
  br i1 %cond2, label %Ret, label %L1
Ret:
  ret void
L1:
  store i32 0, i32* %a
  br label %L42
L2:
  store i32 1, i32* %a
  br label %L42
L42:
  call void @foo()
  br label %Ret
}
