; RUN: not llubi --verbose < %s 2>&1 | FileCheck %s

target datalayout = "e-p:64:64:64"

define void @main() {
  %alloc = alloca i32
  %addr = ptrtoaddr ptr %alloc to i64
  %cast = inttoptr i64 %addr to ptr
  store i32 0, ptr %cast
  ret void
}

; CHECK: Entering function: main
; CHECK-NEXT:   %alloc = alloca i32, align 4 => ptr 0x8 [alloc]
; CHECK-NEXT:   %addr = ptrtoaddr ptr %alloc to i64 => i64 8
; CHECK-NEXT:   %cast = inttoptr i64 %addr to ptr => ptr 0x8 [nullary]
; CHECK-NEXT: Stacktrace:
; CHECK-NEXT: #0   store i32 0, ptr %cast, align 4 at @main {{.*}}
; CHECK-NEXT: Immediate UB detected: Invalid memory access via a pointer with nullary provenance.
; CHECK-NEXT: error: Execution of function 'main' failed.
