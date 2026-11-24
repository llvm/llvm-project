; RUN: opt < %s -passes='csan-module,csan-module' -S | FileCheck %s

; CHECK: @llvm.used = appending global [1 x ptr] [ptr @csan.module_ctor]
; CHECK: @llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 0, ptr @csan.module_ctor, ptr null }]

define void @f() sanitize_concurrency {
  ret void
}

; CHECK-LABEL: define internal void @csan.module_ctor()
; CHECK: call void @__csan_init()
; CHECK: !{i32 4, !"nosanitize_concurrency", i32 1}
