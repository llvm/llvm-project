; RUN: not llvm-as < %s -o /dev/null 2>&1 | FileCheck %s

declare ptr @llvm.dereferenceable(ptr, i64 immarg)

define void @transpose(ptr %p, i64 %x) {
; CHECK: immarg operand has non-immediate parameter
  %d.0 = call ptr @llvm.dereferenceable(ptr %p, i64 4)
  %d.1 = call ptr @llvm.dereferenceable(ptr %p, i64 %x)
  ret void
}

