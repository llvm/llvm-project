; RUN: not llvm-as < %s 2>&1 | FileCheck %s

; CHECK: base element of getelementptr must be sized

%myTy = type { %myTy }
define void @foo(ptr %p){
  getelementptr %myTy, ptr %p, i64 1
  ret void
}
