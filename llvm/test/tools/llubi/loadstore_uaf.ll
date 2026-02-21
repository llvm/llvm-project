; RUN: sed 's/OP1/store i32 0/g' %s | not llubi --verbose 2>&1 | FileCheck %s
; RUN: sed 's/OP1/%res = load i32/g' %s | not llubi --verbose 2>&1 | FileCheck %s

define ptr @stack_object() {
  %alloc = alloca i32
  ret ptr %alloc
}

define void @main() {
  %alloc = call ptr @stack_object()
  OP1, ptr %alloc
  ret void
}
; CHECK: Entering function: main
; CHECK-NEXT: Entering function: stack_object
; CHECK-NEXT:   %alloc = alloca i32, align 4 => ptr 0x8 [alloc]
; CHECK-NEXT:   ret ptr %alloc
; CHECK-NEXT: Exiting function: stack_object
; CHECK-NEXT:   %alloc = call ptr @stack_object() => ptr 0x8 [dangling]
; CHECK-NEXT: Immediate UB detected: Try to access a dead memory object.
; CHECK-NEXT: error: Execution of function 'main' failed.
