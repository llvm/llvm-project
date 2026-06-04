; To test that asan does not break the musttail call contract.
;
; RUN: opt < %s -passes=asan -S | FileCheck %s

define internal i32 @foo(ptr %p) sanitize_address {
  %rv = load i32, ptr %p
  ret i32 %rv
}

declare void @alloca_test_use(ptr)
define i32 @call_foo(ptr %a) sanitize_address {
  %x = alloca [10 x i8], align 1
  call void @alloca_test_use(ptr %x)
  %r = musttail call i32 @foo(ptr %a)
  ret i32 %r
}

; CHECK-LABEL:  define i32 @call_foo(ptr %a) 
; CHECK:          %r = musttail call i32 @foo(ptr %a)
; CHECK-NEXT:     ret i32 %r


define i32 @call_foo_cast(ptr %a) sanitize_address {
  %x = alloca [10 x i8], align 1
  call void @alloca_test_use(ptr %x)
  %r = musttail call i32 @foo(ptr %a)
  %t = bitcast i32 %r to i32
  ret i32 %t
}

; CHECK-LABEL:  define i32 @call_foo_cast(ptr %a)
; CHECK:          %r = musttail call i32 @foo(ptr %a)
; CHECK-NEXT:     %t = bitcast i32 %r to i32
; CHECK-NEXT:     ret i32 %t
