; RUN: opt -passes=globalopt -S -o - < %s | FileCheck %s

; Calls to `noipa` functions in constructors prevent evaluation.

; CHECK: @object = local_unnamed_addr global i32 0
@object = local_unnamed_addr global i32 0
define void @ctor() {
  store i32 4, ptr @object
  %a = bitcast ptr @object to ptr
  call void @test(ptr %a)
  ret void
}

define void @test(ptr %ptr) noipa {
  ret void
}

@llvm.global_ctors = appending constant [1 x { i32, ptr, ptr }] [ { i32, ptr, ptr } { i32 65535, ptr @ctor, ptr null } ]
