; RUN: llubi --verbose < %s 2>&1 | FileCheck %s

target datalayout = "e-p:64:64:64"

define void @main() {
  %p = alloca i32
  %i = ptrtoint ptr %p to i64
  %a = ptrtoaddr ptr %p to i64
  %p2 = inttoptr i64 %a to ptr
  store i32 0, ptr %p2
  ret void
}

; CHECK: Entering function: main
; CHECK-NEXT:   %p = alloca i32, align 4 => ptr 0x8 [p]
; CHECK-NEXT:   %i = ptrtoint ptr %p to i64 => i64 8
; CHECK-NEXT:   %a = ptrtoaddr ptr %p to i64 => i64 8
; CHECK-NEXT:   %p2 = inttoptr i64 %a to ptr => ptr 0x8 [wildcard]
; CHECK-NEXT:   store i32 0, ptr %p2, align 4
; CHECK-NEXT:   ret void
; CHECK-NEXT: Exiting function: main
