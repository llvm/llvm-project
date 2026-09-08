; RUN: llc -mtriple=wasm32-unknown-unknown < %s | FileCheck %s

; A function with the throws (herbception) attribute returns its value with a
; discriminant. On WebAssembly the discriminant is returned as an extra
; multivalue result.

; Success: value in result 0, discriminant (0) in result 1.
define { i64, i1 } @ret_success(i64 %x) #0 {
; CHECK-LABEL: ret_success:
; CHECK:         .functype ret_success (i64) -> (i64, i32)
; CHECK:         local.get 0
; CHECK-NEXT:    i32.const 0
entry:
  %r.i = insertvalue { i64, i1 } poison, i64 %x, 0
  %r = insertvalue { i64, i1 } %r.i, i1 false, 1
  ret { i64, i1 } %r
}

; Error: discriminant is true (1) in result 1.
define { i64, i1 } @ret_error(i64 %x) #0 {
; CHECK-LABEL: ret_error:
; CHECK:         .functype ret_error (i64) -> (i64, i32)
; CHECK:         local.get 0
; CHECK-NEXT:    i32.const 1
entry:
  %r.i = insertvalue { i64, i1 } poison, i64 %x, 0
  %r = insertvalue { i64, i1 } %r.i, i1 true, 1
  ret { i64, i1 } %r
}

; The caller selects on the extra discriminant value.
define i64 @call_and_select(i64 %x) #1 {
; CHECK-LABEL: call_and_select:
; CHECK:         .functype call_and_select (i64) -> (i64)
; CHECK:         call ret_error
; CHECK-NEXT:    local.set 1
; CHECK-NEXT:    local.set 0
; CHECK-NEXT:    i64.const 100
; CHECK-NEXT:    local.get 0
; CHECK-NEXT:    local.get 1
; CHECK-NEXT:    i32.const 1
; CHECK-NEXT:    i32.and
; CHECK-NEXT:    i64.select
entry:
  %c = call { i64, i1 } @ret_error(i64 %x)
  %val = extractvalue { i64, i1 } %c, 0
  %disc = extractvalue { i64, i1 } %c, 1
  %sel = select i1 %disc, i64 100, i64 %val
  ret i64 %sel
}

attributes #0 = { throws }
attributes #1 = { nounwind }
