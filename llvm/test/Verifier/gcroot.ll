; RUN: not llvm-as -o /dev/null %s 2>&1 | FileCheck %s
; PR1633

declare void @llvm.gcroot(ptr, ptr)

define void @caller_must_use_gc() {
  ; CHECK: Enclosing function does not use GC.
  ; CHECK-NEXT: call void @llvm.gcroot(ptr %alloca, ptr null)
  %alloca = alloca ptr
  call void @llvm.gcroot(ptr %alloca, ptr null)
  ret void
}

define void @must_be_alloca() gc "test" {
; CHECK: llvm.gcroot parameter #1 must be an alloca.
; CHECK-NEXT: call void @llvm.gcroot(ptr null, ptr null)
  call void @llvm.gcroot(ptr null, ptr null)
  ret void
}

define void @non_ptr_alloca_null() gc "test" {
  ; CHECK: llvm.gcroot parameter #1 must either be a pointer alloca, or argument #2 must be a non-null constant.
  ; CHECK-NEXT: call void @llvm.gcroot(ptr %alloca, ptr null)
  %alloca = alloca i32
  call void @llvm.gcroot(ptr %alloca, ptr null)
  ret void
}

define void @non_constant_arg1(ptr %arg) gc "test" {
  ; CHECK: llvm.gcroot parameter #2 must be a constant.
  ; CHECK-NEXT: call void @llvm.gcroot(ptr %alloca, ptr %arg)
  %alloca = alloca ptr
  call void @llvm.gcroot(ptr %alloca, ptr %arg)
  ret void
}

define void @non_ptr_alloca_non_null() gc "test" {
; CHECK-NOT: llvm.gcroot parameter
  %alloca = alloca i32
  call void @llvm.gcroot(ptr %alloca, ptr inttoptr (i64 123 to ptr))
  ret void
}

define void @casted_alloca() gc "test" {
; CHECK-NOT: llvm.gcroot parameter
  %alloca = alloca ptr
  call void @llvm.gcroot(ptr %alloca, ptr null)
  ret void
}
