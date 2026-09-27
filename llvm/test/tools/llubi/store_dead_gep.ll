; RUN: not llubi --verbose < %s 2>&1 | FileCheck %s

define void @main() {
  %alloc = alloca [4 x i8], align 1
  call void @llvm.lifetime.start.p0(ptr %alloc)
  store i8 0, ptr %alloc, align 1
  call void @llvm.lifetime.end.p0(ptr %alloc)
  %gep = getelementptr i8, ptr %alloc, i64 1
  store i8 0, ptr %gep, align 1
  ret void
}

; CHECK: Entering function: main
; CHECK-NEXT:   %alloc = alloca [4 x i8], align 1 => ptr 0x8 [alloc (dead)]
; CHECK-NEXT:   call void @llvm.lifetime.start.p0(ptr %alloc)
; CHECK-NEXT:   store i8 0, ptr %alloc, align 1
; CHECK-NEXT:   call void @llvm.lifetime.end.p0(ptr %alloc)
; CHECK-NEXT:   %gep = getelementptr i8, ptr %alloc, i64 1 => ptr 0x9 [alloc + 1 (dead)]
; CHECK-NEXT: Stacktrace:
; CHECK-NEXT: #0   store i8 0, ptr %gep, align 1 at @main <stdin>:9
; CHECK-NEXT: Immediate UB detected: Try to access a dead memory object at address 0x9.
; CHECK-NEXT: error: Execution of function 'main' failed.
