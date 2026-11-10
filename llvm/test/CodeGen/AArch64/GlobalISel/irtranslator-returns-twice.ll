; RUN: llc -mtriple=aarch64-linux-gnu -global-isel -global-isel-abort=1 \
; RUN:   -stop-after=ir-translator %s -o - | FileCheck %s

; Check that the IRTranslator sets exposesReturnsTwice on functions that call
; a returns_twice function.

declare i32 @setjmp(ptr) returns_twice
declare i32 @plain(ptr)
declare i32 @__gxx_personality_v0(...)

; The attribute is on the callee declaration.
define i32 @call_decl_attr(ptr %buf) {
; CHECK-LABEL: name: call_decl_attr
; CHECK:       exposesReturnsTwice: true
  %r = call i32 @setjmp(ptr %buf)
  ret i32 %r
}

; The attribute is only on the call site.
define i32 @call_site_attr(ptr %buf) {
; CHECK-LABEL: name: call_site_attr
; CHECK:       exposesReturnsTwice: true
  %r = call i32 @plain(ptr %buf) returns_twice
  ret i32 %r
}

; The returns_twice call is not in the entry block.
define i32 @call_in_later_block(ptr %buf, i1 %c) {
; CHECK-LABEL: name: call_in_later_block
; CHECK:       exposesReturnsTwice: true
entry:
  br i1 %c, label %then, label %exit

then:
  %r = call i32 @setjmp(ptr %buf)
  br label %exit

exit:
  %p = phi i32 [ 0, %entry ], [ %r, %then ]
  ret i32 %p
}

; An invoke of a returns_twice function also counts.
define i32 @invoke_returns_twice(ptr %buf) personality ptr @__gxx_personality_v0 {
; CHECK-LABEL: name: invoke_returns_twice
; CHECK:       exposesReturnsTwice: true
entry:
  %r = invoke i32 @setjmp(ptr %buf)
          to label %cont unwind label %lpad

cont:
  ret i32 %r

lpad:
  %lp = landingpad { ptr, i32 }
          cleanup
  ret i32 0
}

; No returns_twice call.
define i32 @no_returns_twice(ptr %buf) {
; CHECK-LABEL: name: no_returns_twice
; CHECK:       exposesReturnsTwice: false
  %r = call i32 @plain(ptr %buf)
  ret i32 %r
}
