; RUN: opt -passes=inline -S %s 2>&1 | FileCheck %s
; CHECK-NOT: LLVM ERROR: Broken module found, compilation aborted!
define void @bar(i32 %stride) {
  call void @llvm.matrix.column.major.store.v6f64.i32(<6 x double> zeroinitializer, ptr null, i32 %stride, i1 false, i32 3, i32 2)
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: write)
declare void @llvm.matrix.column.major.store.v6f64.i32(<6 x double>, ptr nocapture writeonly, i32, i1 immarg, i32 immarg, i32 immarg)

define i64 @foo() {
entry:
  unreachable

sink.call:                                        ; No predecessors!
  call void @bar(i32 0)
  ret i64 0
}