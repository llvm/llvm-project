; RUN: opt -passes=prof-verify %s -o - 2>&1 | FileCheck %s
@llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 0, ptr @ctor, ptr null }]
@llvm.global_dtors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 0, ptr @dtor, ptr null }]

define internal void @ctor() {
  ret void
}

define internal void @dtor() {
  ret void
}

; CHECK-NOT: Profile verification failed for function {{.+}}: function entry count missing
