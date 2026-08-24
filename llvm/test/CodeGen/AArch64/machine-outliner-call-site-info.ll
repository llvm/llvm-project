; Test how the machine outliner handles calls depending on call site info: a
; call known to have no stack arguments may be outlined non-terminally, while a
; call with stack arguments (or no call site info) may only end an outlined
; function.

; RUN: split-file %s %t

; RUN: llc -mtriple=aarch64-apple-darwin -enable-machine-outliner -emit-codegen-call-site-info < %t/direct.ll | FileCheck %t/direct.ll
; RUN: llc -mtriple=aarch64-apple-darwin -enable-machine-outliner -emit-codegen-call-site-info=false < %t/direct.ll | FileCheck %t/direct.ll --check-prefix=OFF

; RUN: llc -mtriple=aarch64-apple-darwin -enable-machine-outliner -emit-codegen-call-site-info < %t/indirect.ll | FileCheck %t/indirect.ll
; RUN: llc -mtriple=aarch64-apple-darwin -enable-machine-outliner -emit-codegen-call-site-info=false < %t/indirect.ll | FileCheck %t/indirect.ll --check-prefix=OFF

; RUN: llc -mtriple=aarch64-apple-darwin -enable-machine-outliner -emit-codegen-call-site-info < %t/stack.ll | FileCheck %t/stack.ll
; RUN: llc -mtriple=aarch64-apple-darwin -enable-machine-outliner -emit-codegen-call-site-info=false < %t/stack.ll | FileCheck %t/stack.ll

; RUN: llc -mtriple=aarch64-apple-darwin -enable-machine-outliner -emit-codegen-call-site-info < %t/indirect-stack.ll | FileCheck %t/indirect-stack.ll
; RUN: llc -mtriple=aarch64-apple-darwin -enable-machine-outliner -emit-codegen-call-site-info=false < %t/indirect-stack.ll | FileCheck %t/indirect-stack.ll

;--- direct.ll

@g = external global i32

declare void @no_stack_args()

; CHECK-LABEL: _fn1:
; CHECK:       bl [[OUTLINED:_OUTLINED_FUNCTION_[0-9]+]]

; CHECK:       [[OUTLINED]]:
; CHECK:       bl _no_stack_args

; OFF-LABEL: _fn1:
; OFF:        bl [[OUTLINED:_OUTLINED_FUNCTION_[0-9]+]]

; OFF:        [[OUTLINED]]:
; OFF:        b _no_stack_args

define void @fn1() minsize {
entry:
  store volatile i32 1, ptr @g
  call void @no_stack_args()
  store volatile i32 2, ptr @g
  ret void
}

define void @fn2() minsize {
entry:
  store volatile i32 1, ptr @g
  call void @no_stack_args()
  store volatile i32 2, ptr @g
  ret void
}

define void @fn3() minsize {
entry:
  store volatile i32 1, ptr @g
  call void @no_stack_args()
  store volatile i32 2, ptr @g
  ret void
}

;--- indirect.ll

@g = external global i32

; CHECK-LABEL: _indirect1:
; CHECK:       bl [[OUTLINED:_OUTLINED_FUNCTION_[0-9]+]]

; CHECK:       [[OUTLINED]]:
; CHECK:       blr x{{[0-9]+}}

; OFF-LABEL: _indirect1:
; OFF:        bl [[OUTLINED:_OUTLINED_FUNCTION_[0-9]+]]

; OFF:        [[OUTLINED]]:
; OFF:        br x{{[0-9]+}}

define void @indirect1(ptr %fptr) minsize {
entry:
  %v = load volatile i32, ptr @g
  call void %fptr()
  %v2 = load volatile i32, ptr @g
  ret void
}

define void @indirect2(ptr %fptr) minsize {
entry:
  %v = load volatile i32, ptr @g
  call void %fptr()
  %v2 = load volatile i32, ptr @g
  ret void
}

define void @indirect3(ptr %fptr) minsize {
entry:
  %v = load volatile i32, ptr @g
  call void %fptr()
  %v2 = load volatile i32, ptr @g
  ret void
}

;--- stack.ll

@g = external global i32

declare void @has_stack_args(i64, i64, i64, i64, i64, i64, i64, i64, i64)

; CHECK-LABEL: _stack1:
; CHECK:       bl [[OUTLINED:_OUTLINED_FUNCTION_[0-9]+]]

; CHECK:       [[OUTLINED]]:
; CHECK:       b _has_stack_args

define void @stack1() minsize {
entry:
  store volatile i32 1, ptr @g
  call void @has_stack_args(i64 0, i64 1, i64 2, i64 3, i64 4, i64 5, i64 6, i64 7, i64 8)
  store volatile i32 2, ptr @g
  ret void
}

define void @stack2() minsize {
entry:
  store volatile i32 1, ptr @g
  call void @has_stack_args(i64 0, i64 1, i64 2, i64 3, i64 4, i64 5, i64 6, i64 7, i64 8)
  store volatile i32 2, ptr @g
  ret void
}

define void @stack3() minsize {
entry:
  store volatile i32 1, ptr @g
  call void @has_stack_args(i64 0, i64 1, i64 2, i64 3, i64 4, i64 5, i64 6, i64 7, i64 8)
  store volatile i32 2, ptr @g
  ret void
}

;--- indirect-stack.ll

@g = external global i32

; CHECK-LABEL: _indirect_stack1:
; CHECK:       bl [[OUTLINED:_OUTLINED_FUNCTION_[0-9]+]]

; CHECK:       [[OUTLINED]]:
; CHECK:       br x{{[0-9]+}}

define void @indirect_stack1(ptr %fptr) minsize {
entry:
  %v = load volatile i32, ptr @g
  call void %fptr(i64 0, i64 1, i64 2, i64 3, i64 4, i64 5, i64 6, i64 7, i64 8)
  %v2 = load volatile i32, ptr @g
  ret void
}

define void @indirect_stack2(ptr %fptr) minsize {
entry:
  %v = load volatile i32, ptr @g
  call void %fptr(i64 0, i64 1, i64 2, i64 3, i64 4, i64 5, i64 6, i64 7, i64 8)
  %v2 = load volatile i32, ptr @g
  ret void
}

define void @indirect_stack3(ptr %fptr) minsize {
entry:
  %v = load volatile i32, ptr @g
  call void %fptr(i64 0, i64 1, i64 2, i64 3, i64 4, i64 5, i64 6, i64 7, i64 8)
  %v2 = load volatile i32, ptr @g
  ret void
}
