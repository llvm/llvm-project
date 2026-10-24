; RUN: opt -S -mtriple=bpfel -passes=bpf-func-table %s | FileCheck %s
; RUN: llc -mtriple=bpfel -mcpu=v4 -mattr=+func-table %s -o - | FileCheck --check-prefix=ASM %s
;
; The kernel calls what is passed to a helper or to a kfunc as a callback.
; It wants the function and not its index in the table.

declare i32 @bpf_wq_set_callback(ptr, ptr, i32)

define internal i64 @cb(i64 %i, ptr %ctx) {
  ret i64 0
}

define internal void @take(ptr %f) noinline {
  %r = call i64 %f(i64 0, ptr null)
  ret void
}

define i64 @helper(ptr %ctx) {
; CHECK-LABEL: @helper(
; CHECK:       call i64 inttoptr (i64 181 to ptr)(i32 10, ptr @cb, ptr %ctx, i64 0)
; ASM-LABEL:   helper:
; ASM:         r2 = cb ll
; ASM:         call 181
  %r = call i64 inttoptr (i64 181 to ptr)(i32 10, ptr @cb, ptr %ctx, i64 0)
  ret i64 %r
}

define i32 @kfunc(ptr %wq) {
; CHECK-LABEL: @kfunc(
; CHECK:       call i32 @bpf_wq_set_callback(ptr %wq, ptr @cb, i32 0)
; ASM-LABEL:   kfunc:
; ASM:         r2 = cb ll
; ASM:         call bpf_wq_set_callback
  %r = call i32 @bpf_wq_set_callback(ptr %wq, ptr @cb, i32 0)
  ret i32 %r
}

; Passed to a function of the program it is an index.
define void @own() {
; CHECK-LABEL: @own(
; CHECK:       call void @take(ptr inttoptr (i64 1 to ptr))
; ASM-LABEL:   own:
; ASM:         r1 = 1
; ASM:         call take
  call void @take(ptr @cb)
  ret void
}
