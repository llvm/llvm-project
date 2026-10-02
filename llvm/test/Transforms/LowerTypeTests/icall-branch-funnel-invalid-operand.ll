; RUN: not opt -S -passes=lowertypetests %s 2>&1 | FileCheck %s
; CHECK: LLVM ERROR: Expected branch funnel operand to be a defined global value with type metadata

target triple = "x86_64--"

@g1 = external constant i32
@g2 = external constant i32

define void @jt2(...) {
  musttail call void (...) @llvm.icall.branch.funnel(ptr null, ptr @g1, ptr null, ptr @g2, ptr null, ...)
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind willreturn
declare void @llvm.icall.branch.funnel(...) #0

attributes #0 = { nocallback nofree nosync nounwind willreturn }
