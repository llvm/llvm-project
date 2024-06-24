;; Run opt with asan to trigger `appendToGlobalArray` call which should update uses of `llvm.global_ctors`
; RUN: opt -passes=asan -S %s -o - | FileCheck %s
; CHECK: @llvm.global_ctors = appending global [2 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr ptrauth (ptr @foo, i32 0, i64 55764, ptr getelementptr inbounds ([1 x { i32, ptr, ptr }], ptr @llvm.global_ctors, i32 0, i32 0, i32 1)), ptr null }, { i32, ptr, ptr } { i32 1, ptr @asan.module_ctor, ptr @asan.module_ctor }]

@llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr ptrauth (ptr @foo, i32 0, i64 55764, ptr getelementptr inbounds ([1 x { i32, ptr, ptr }], ptr @llvm.global_ctors, i32 0, i32 0, i32 1)), ptr null }]

define void @foo() {
  ret void
}
