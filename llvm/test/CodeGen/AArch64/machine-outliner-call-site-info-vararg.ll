; A call to a vararg function that passes no variadic arguments has no stack
; arguments, so with call site info it can be outlined non-terminally. This is
; preferred over the callee's frame info, and it also applies to external
; vararg callees whose definition is not visible.

; RUN: split-file %s %t

; RUN: llc -mtriple=aarch64-apple-darwin -enable-machine-outliner -emit-codegen-call-site-info < %t/local-no-stack.ll | FileCheck %t/local-no-stack.ll
; RUN: llc -mtriple=aarch64-apple-darwin -enable-machine-outliner -emit-codegen-call-site-info=false < %t/local-no-stack.ll | FileCheck %t/local-no-stack.ll --check-prefix=OFF

; RUN: llc -mtriple=aarch64-apple-darwin -enable-machine-outliner -emit-codegen-call-site-info < %t/local-on-stack.ll | FileCheck %t/local-on-stack.ll
; RUN: llc -mtriple=aarch64-apple-darwin -enable-machine-outliner -emit-codegen-call-site-info=false < %t/local-on-stack.ll | FileCheck %t/local-on-stack.ll

; RUN: llc -mtriple=aarch64-apple-darwin -enable-machine-outliner -emit-codegen-call-site-info < %t/external-no-stack.ll | FileCheck %t/external-no-stack.ll
; RUN: llc -mtriple=aarch64-apple-darwin -enable-machine-outliner -emit-codegen-call-site-info=false < %t/external-no-stack.ll | FileCheck %t/external-no-stack.ll --check-prefix=OFF

; RUN: llc -mtriple=aarch64-apple-darwin -enable-machine-outliner -emit-codegen-call-site-info < %t/external-on-stack.ll | FileCheck %t/external-on-stack.ll
; RUN: llc -mtriple=aarch64-apple-darwin -enable-machine-outliner -emit-codegen-call-site-info=false < %t/external-on-stack.ll | FileCheck %t/external-on-stack.ll

;--- local-no-stack.ll

@g = external global i32

declare void @llvm.va_start(ptr)
declare void @llvm.va_end(ptr)

define internal void @local_vararg(i32 %n, ...) minsize {
entry:
  %ap = alloca ptr
  call void @llvm.va_start(ptr %ap)
  %v = va_arg ptr %ap, i32
  store volatile i32 %v, ptr @g
  call void @llvm.va_end(ptr %ap)
  ret void
}

; CHECK-LABEL: _call_local_no_stack_1:
; CHECK:       bl [[OUTLINED:_OUTLINED_FUNCTION_[0-9]+]]

; CHECK:       [[OUTLINED]]:
; CHECK:       bl _local_vararg

; OFF-LABEL: _call_local_no_stack_1:
; OFF:        bl [[OUTLINED:_OUTLINED_FUNCTION_[0-9]+]]

; OFF:        [[OUTLINED]]:
; OFF:        b _local_vararg

define void @call_local_no_stack_1() minsize {
  store volatile i32 1, ptr @g
  call void (i32, ...) @local_vararg(i32 7)
  store volatile i32 2, ptr @g
  ret void
}
define void @call_local_no_stack_2() minsize {
  store volatile i32 1, ptr @g
  call void (i32, ...) @local_vararg(i32 7)
  store volatile i32 2, ptr @g
  ret void
}
define void @call_local_no_stack_3() minsize {
  store volatile i32 1, ptr @g
  call void (i32, ...) @local_vararg(i32 7)
  store volatile i32 2, ptr @g
  ret void
}

;--- local-on-stack.ll

@g = external global i32

declare void @llvm.va_start(ptr)
declare void @llvm.va_end(ptr)

define internal void @local_vararg(i32 %n, ...) minsize {
entry:
  %ap = alloca ptr
  call void @llvm.va_start(ptr %ap)
  %v = va_arg ptr %ap, i32
  store volatile i32 %v, ptr @g
  call void @llvm.va_end(ptr %ap)
  ret void
}

; CHECK-LABEL: _call_local_on_stack_1:
; CHECK:       bl [[OUTLINED:_OUTLINED_FUNCTION_[0-9]+]]

; CHECK:       [[OUTLINED]]:
; CHECK:       b _local_vararg

define void @call_local_on_stack_1() minsize {
  store volatile i32 1, ptr @g
  call void (i32, ...) @local_vararg(i32 7, i64 0, i64 1, i64 2, i64 3, i64 4, i64 5, i64 6, i64 7, i64 8)
  store volatile i32 2, ptr @g
  ret void
}
define void @call_local_on_stack_2() minsize {
  store volatile i32 1, ptr @g
  call void (i32, ...) @local_vararg(i32 7, i64 0, i64 1, i64 2, i64 3, i64 4, i64 5, i64 6, i64 7, i64 8)
  store volatile i32 2, ptr @g
  ret void
}
define void @call_local_on_stack_3() minsize {
  store volatile i32 1, ptr @g
  call void (i32, ...) @local_vararg(i32 7, i64 0, i64 1, i64 2, i64 3, i64 4, i64 5, i64 6, i64 7, i64 8)
  store volatile i32 2, ptr @g
  ret void
}

;--- external-no-stack.ll

@g = external global i32

declare void @external_vararg(i32, ...)

; CHECK-LABEL: _call_external_no_stack_1:
; CHECK:       bl [[OUTLINED:_OUTLINED_FUNCTION_[0-9]+]]

; CHECK:       [[OUTLINED]]:
; CHECK:       bl _external_vararg

; OFF-LABEL: _call_external_no_stack_1:
; OFF:        bl [[OUTLINED:_OUTLINED_FUNCTION_[0-9]+]]

; OFF:        [[OUTLINED]]:
; OFF:        b _external_vararg

define void @call_external_no_stack_1() minsize {
  store volatile i32 1, ptr @g
  call void (i32, ...) @external_vararg(i32 7)
  store volatile i32 2, ptr @g
  ret void
}
define void @call_external_no_stack_2() minsize {
  store volatile i32 1, ptr @g
  call void (i32, ...) @external_vararg(i32 7)
  store volatile i32 2, ptr @g
  ret void
}
define void @call_external_no_stack_3() minsize {
  store volatile i32 1, ptr @g
  call void (i32, ...) @external_vararg(i32 7)
  store volatile i32 2, ptr @g
  ret void
}

;--- external-on-stack.ll

@g = external global i32

declare void @external_vararg(i32, ...)

; CHECK-LABEL: _call_external_on_stack_1:
; CHECK:       bl [[OUTLINED:_OUTLINED_FUNCTION_[0-9]+]]

; CHECK:       [[OUTLINED]]:
; CHECK:       b _external_vararg

define void @call_external_on_stack_1() minsize {
  store volatile i32 1, ptr @g
  call void (i32, ...) @external_vararg(i32 7, i64 0, i64 1, i64 2, i64 3, i64 4, i64 5, i64 6, i64 7, i64 8)
  store volatile i32 2, ptr @g
  ret void
}
define void @call_external_on_stack_2() minsize {
  store volatile i32 1, ptr @g
  call void (i32, ...) @external_vararg(i32 7, i64 0, i64 1, i64 2, i64 3, i64 4, i64 5, i64 6, i64 7, i64 8)
  store volatile i32 2, ptr @g
  ret void
}
define void @call_external_on_stack_3() minsize {
  store volatile i32 1, ptr @g
  call void (i32, ...) @external_vararg(i32 7, i64 0, i64 1, i64 2, i64 3, i64 4, i64 5, i64 6, i64 7, i64 8)
  store volatile i32 2, ptr @g
  ret void
}
