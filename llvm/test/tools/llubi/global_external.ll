; RUN: llubi --verbose < %s 2>&1 | FileCheck %s

target datalayout = "e-p:64:64:64-i32:32:32"

@external = external global i32
@external_ptr = global ptr @external

define void @main() {
  %ptr = load ptr, ptr @external_ptr
  store i32 42, ptr @external
  %loaded = load i32, ptr @external
  ret void
}

; CHECK: Entering function: main
; CHECK-NEXT:   %ptr = load ptr, ptr @external_ptr, align 8 => ptr 0x8 [@external]
; CHECK-NEXT:   store i32 42, ptr @external, align 4
; CHECK-NEXT:   %loaded = load i32, ptr @external, align 4 => i32 42
; CHECK-NEXT:   ret void
; CHECK-NEXT: Exiting function: main
