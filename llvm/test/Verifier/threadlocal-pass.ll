; RUN: opt -passes=verify -S < %s | FileCheck %s

@var = thread_local global i32 0
@alias = thread_local alias i32, ptr @var

; CHECK-LABEL: @should_pass
define void @should_pass() {
  %p0 = call ptr @llvm.threadlocal.address(ptr @var)
  store i32 42, ptr %p0, align 4
  %p1 = call ptr @llvm.threadlocal.address(ptr @alias)
  store i32 13, ptr %p1, align 4
  ret void
}
