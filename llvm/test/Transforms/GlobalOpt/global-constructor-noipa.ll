; RUN: opt -passes=globalopt -S -o - < %s | FileCheck %s

; Calls to `noipa` functions in constructors prevent evaluation.

; CHECK: @object = local_unnamed_addr global i32 4
@object = local_unnamed_addr global i32 0

define void @ctor() {
  store i32 4, ptr @object
  %a = bitcast ptr @object to ptr
  call void @test(ptr %a)
  ret void
}

define void @test(ptr %ptr) {
  ret void
}

; CHECK: @object_noipa = local_unnamed_addr global i32 0
@object_noipa = local_unnamed_addr global i32 0

define void @ctor_noipa() {
  store i32 4, ptr @object_noipa
  %a = bitcast ptr @object_noipa to ptr
  call void @test_noipa(ptr %a)
  ret void
}

define void @test_noipa(ptr %ptr) noipa {
  ret void
}

@llvm.global_ctors = appending constant
  [2 x { i32, ptr, ptr }]
  [ { i32, ptr, ptr } { i32 65535, ptr @ctor, ptr null },
    { i32, ptr, ptr } { i32 65535, ptr @ctor_noipa, ptr null } ]
