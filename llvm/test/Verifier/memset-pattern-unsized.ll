; RUN: not opt -passes=verify < %s 2>&1 | FileCheck %s

; CHECK: unsized types cannot be used as memset patterns

define void @bar(ptr %P, target("foo") %value) {
  call void @llvm.experimental.memset.pattern.p0.s_s.i32.0(ptr %P, target("foo") %value, i32 4, i1 false)
  ret void
}
