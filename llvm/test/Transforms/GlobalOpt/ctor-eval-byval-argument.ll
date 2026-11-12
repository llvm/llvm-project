; RUN: opt -passes=globalopt -S < %s | FileCheck %s

; CHECK: @llvm.global_ctors = appending {{.*}}@ctor
; CHECK-LABEL: define internal {{.*}}void @ctor(

%struct.S = type { i64, i64, i64 }

@g = global %struct.S zeroinitializer
@seen = global i64 0
@llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr @ctor, ptr null }]

define internal void @f(ptr byval(%struct.S) %s) {
  store i64 1, ptr @g
  %v = load i64, ptr %s
  store i64 %v, ptr @seen
  ret void
}

define internal void @ctor() {
  call void @f(ptr byval(%struct.S) @g)
  ret void
}
