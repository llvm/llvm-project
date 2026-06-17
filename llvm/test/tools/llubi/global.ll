; RUN: llubi --verbose < %s 2>&1 | FileCheck %s

target datalayout = "e-p:64:64:64-i32:32:32"

@value = global i32 41
@value_ptr = global ptr @value
@aggregate = global { i32, [2 x i8] } { i32 7, [2 x i8] c"hi" }

define void @main() {
  %initial = load i32, ptr @value
  store i32 42, ptr @value
  %ptr = load ptr, ptr @value_ptr
  %updated = load i32, ptr %ptr
  %aggregate_value = load { i32, [2 x i8] }, ptr @aggregate
  ret void
}

; CHECK: Entering function: main
; CHECK-NEXT:   %initial = load i32, ptr @value, align 4 => i32 41
; CHECK-NEXT:   store i32 42, ptr @value, align 4
; CHECK-NEXT:   %ptr = load ptr, ptr @value_ptr, align 8 => ptr 0x8 [@value]
; CHECK-NEXT:   %updated = load i32, ptr %ptr, align 4 => i32 42
; CHECK-NEXT:   %aggregate_value = load { i32, [2 x i8] }, ptr @aggregate, align 4 => { i32 7, { i8 104, i8 105 } }
; CHECK-NEXT:   ret void
; CHECK-NEXT: Exiting function: main
