; Test that the SPIR-V constructor/destructor lowering pass does not touch
; logical SPIR-V (shader) modules.

; RUN: opt -S -passes=spirv-lower-ctor-dtor -mtriple=spirv1.6-unknown-vulkan1.3 < %s | FileCheck %s

define void @my_ctor() {
entry:
  ret void
}

define void @my_dtor() {
entry:
  ret void
}

@llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [
  { i32, ptr, ptr } { i32 65535, ptr @my_ctor, ptr null }
]

@llvm.global_dtors = appending global [1 x { i32, ptr, ptr }] [
  { i32, ptr, ptr } { i32 65535, ptr @my_dtor, ptr null }
]

; CHECK: @llvm.global_ctors = appending global
; CHECK: @llvm.global_dtors = appending global
; CHECK-NOT: __init_array
; CHECK-NOT: __fini_array
; CHECK-NOT: spirv$device$init
; CHECK-NOT: spirv$device$fini
