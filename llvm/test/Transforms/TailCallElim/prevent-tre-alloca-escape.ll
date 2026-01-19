; RUN: opt -passes=tailcallelim -S < %s | FileCheck %s

; Test that Tail Call Elimination correctly identifies when an alloca's
; address is stored in another alloca and subsequently escapes.
; The analysis should recognize that if the stored address is loaded back
; and passed to an opaque function, the original alloca is no longer safe
; for TRE because its lifetime could be extended beyond the current frame.

declare void @opaque(ptr)

define i32 @prevent_tre_on_load_escape(i32 %n, i32 %acc) {
; CHECK-LABEL: @prevent_tre_on_load_escape
; CHECK-NOT: tailrecurse:
; CHECK: %res = call i32 @prevent_tre_on_load_escape
entry:
  %local_var = alloca i32
  %ptr_storage = alloca ptr

  ; Store the address of the local variable into a temporary storage alloca.
  store ptr %local_var, ptr %ptr_storage

  ; Access the stored address and pass it to an external function.
  ; This load and subsequent call make %local_var "escape".
  %addr = load ptr, ptr %ptr_storage
  call void @opaque(ptr %addr)

  ; The flow-sensitive analysis should detect that %local_var escapes
  ; through %ptr_storage. Any recursive call in this function must NOT
  ; be transformed into a loop (TRE).

  %cond = icmp eq i32 %n, 0
  br i1 %cond, label %exit, label %recurse

recurse:
  %n_dec = sub i32 %n, 1
  %acc_inc = add i32 %acc, 1

  ; TRE must NOT be applied here because the address of %local_var
  ; has escaped and could be accessed by @opaque during the next iterations.
  %res = call i32 @prevent_tre_on_load_escape(i32 %n_dec, i32 %acc_inc)
  ret i32 %res

exit:
  ret i32 %acc
}
