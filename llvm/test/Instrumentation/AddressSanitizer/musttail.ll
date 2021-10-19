; To test that asan does not break the musttail call contract.
;
; RUN: opt < %s -passes='asan-pipeline' -S | FileCheck %s

define internal i32 @foo(i32* %p) sanitize_address {
  %rv = load i32, i32* %p
  ret i32 %rv
}

declare void @alloca_test_use([10 x i8]*)
define i32 @call_foo(i32* %a) sanitize_address {
  %x = alloca [10 x i8], align 1
  call void @alloca_test_use([10 x i8]* %x)
  %r = musttail call i32 @foo(i32* %a)
  ret i32 %r
}

; CHECK-LABEL:  define i32 @call_foo(i32* %a) 
; CHECK:          %r = musttail call i32 @foo(i32* %a)
; CHECK-NEXT:     ret i32 %r


define i32 @call_foo_cast(i32* %a) sanitize_address {
  %x = alloca [10 x i8], align 1
  call void @alloca_test_use([10 x i8]* %x)
  %r = musttail call i32 @foo(i32* %a)
  %t = bitcast i32 %r to i32
  ret i32 %t
}

; CHECK-LABEL:  define i32 @call_foo_cast(i32* %a)
; CHECK:          %r = musttail call i32 @foo(i32* %a)
; CHECK-NEXT:     %t = bitcast i32 %r to i32
; CHECK-NEXT:     ret i32 %t
