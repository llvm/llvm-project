; RUN: opt -passes=print-predicateinfo < %s 2>&1 | FileCheck %s

declare void @foo()
declare void @llvm.assume(i1)

define void @bar(ptr %p) {
entry:
; CHECK-LABEL: @bar
  br label %end

unreachable1:
  %v1 = load i32, ptr %p, align 4
  %c1 = icmp eq i32 %v1, 0
  call void @llvm.assume(i1 %c1)
  br label %unreachable2

unreachable2:
  %v2 = load i32, ptr %p, align 4
  %c2 = icmp eq i32 %v2, 0
  call void @llvm.assume(i1 %c2)
  br label %end

end:
  ret void
}
