; RUN: sed 's/OP1/store i32 0/g' %s | not llubi --verbose 2>&1 | FileCheck %s
; RUN: sed 's/OP1/%res = load i32/g' %s | not llubi --verbose 2>&1 | FileCheck %s

define void @main() {
  %alloc = alloca [2 x i32]
  %gep = getelementptr inbounds [2 x i32], ptr %alloc, i64 0, i64 2
  OP1, ptr %gep
  ret void
}
; CHECK: Entering function: main
; CHECK-NEXT: %alloc = alloca [2 x i32], align 4 => ptr 0x8 [alloc]
; CHECK-NEXT: %gep = getelementptr inbounds [2 x i32], ptr %alloc, i64 0, i64 2 => ptr 0x10 [alloc + 8]
; CHECK-NEXT: Stacktrace:
; CHECK-NEXT: #0   {{store i32 0|%res = load i32}}, ptr %gep, align 4 at @main
; CHECK-NEXT: Immediate UB detected: Memory access is out of bounds. Accessed size: 4, Address: 0x10, Object base: 0x8, Object size: 8.
; CHECK-NEXT: error: Execution of function 'main' failed.
