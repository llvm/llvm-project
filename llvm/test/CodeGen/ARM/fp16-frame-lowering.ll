; RUN: llc < %s -mtriple armv8a--none-eabi -mattr=+fullfp16             | FileCheck %s
; RUN: llc < %s -mtriple armv8a--none-eabi -mattr=+fullfp16,+thumb-mode | FileCheck %s

; Check that frame lowering for the fp16 instructions works correctly with
; negative offsets (which happens when using the frame pointer).

define void @foo(i32 %count) {
entry:
  %half_alloca = alloca half, align 2
; CHECK: vstr.16 {{s[0-9]+}}, [{{r[0-9]+}}, #-10]
  store half 0.0, ptr %half_alloca
  call void @bar(ptr %half_alloca)

  ; A variable-sized alloca to force the above store to use the frame pointer
  ; instead of the stack pointer, and so need a negative offset.
  %var_alloca = alloca i32, i32 %count
  call void @baz(ptr %var_alloca)
  ret void
}

declare void @bar(ptr)
declare void @baz(ptr)
