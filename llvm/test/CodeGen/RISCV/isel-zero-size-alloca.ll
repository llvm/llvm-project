; RUN: llc -mtriple=riscv64 -stop-after=finalize-isel < %s | FileCheck %s
; RUN: llc -mtriple=riscv64 -global-isel -stop-after=ir-translator < %s | FileCheck %s

; A zero-size alloca gets a one-byte frame object, so it is updated to allocate
; that byte: the frame object links back to it and alias analysis describes the
; object by it. The size can be zero from the allocated type, the array
; operand, or both.
;
; The updated allocas are printed in the module preamble.

; CHECK:       @zero_size_type
; CHECK:         %a = alloca i8, align 1
; CHECK:       @empty_struct
; CHECK:         %a = alloca i8, align 1
; CHECK:       @zero_array_operand
; CHECK:         %a = alloca i8, align 1
; CHECK:       @zero_size_type_and_array_operand
; CHECK:         %a = alloca i8, align 1
; The operand keeps the type it had.
; CHECK:       @i64_array_operand
; CHECK:         %a = alloca i8, i64 1, align 1

define void @zero_size_type() nounwind {
; CHECK-LABEL: name: zero_size_type
; CHECK:       stack:
; CHECK-NEXT:    - { id: 0, name: a, type: default, offset: 0, size: 1
  %a = alloca [0 x i8], align 1
  call void @use(ptr %a)
  ret void
}

define void @empty_struct() nounwind {
; CHECK-LABEL: name: empty_struct
; CHECK:       stack:
; CHECK-NEXT:    - { id: 0, name: a, type: default, offset: 0, size: 1
  %a = alloca {}, align 1
  call void @use(ptr %a)
  ret void
}

define void @zero_array_operand() nounwind {
; CHECK-LABEL: name: zero_array_operand
; CHECK:       stack:
; CHECK-NEXT:    - { id: 0, name: a, type: default, offset: 0, size: 1
  %a = alloca i8, i32 0, align 1
  call void @use(ptr %a)
  ret void
}

define void @zero_size_type_and_array_operand() nounwind {
; CHECK-LABEL: name: zero_size_type_and_array_operand
; CHECK:       stack:
; CHECK-NEXT:    - { id: 0, name: a, type: default, offset: 0, size: 1
  %a = alloca [0 x i8], i32 4, align 1
  call void @use(ptr %a)
  ret void
}

define void @i64_array_operand() nounwind {
; CHECK-LABEL: name: i64_array_operand
; CHECK:       stack:
; CHECK-NEXT:    - { id: 0, name: a, type: default, offset: 0, size: 1
  %a = alloca [0 x i8], i64 7, align 1
  call void @use(ptr %a)
  ret void
}

declare void @use(ptr)
