; RUN: llvm-link %s %p/Inputs/byref-type-input.ll -S | FileCheck %s

%a = type { i64 }
%struct = type { i32, i8 }

; CHECK-LABEL: define void @f(ptr byref(%a) %0)
define void @f(ptr byref(%a)) {
  ret void
}

; CHECK-LABEL: define void @bar(
; CHECK: call void @foo(ptr byref(%struct) %ptr)
define void @bar() {
  %ptr = alloca %struct
  call void @foo(ptr byref(%struct) %ptr)
  ret void
}

; CHECK-LABEL: define void @g(ptr byref(%a) %0)

; CHECK-LABEL: define void @foo(ptr byref(%struct) %a)
; CHECK-NEXT:   call void @baz(ptr byref(%struct) %a)
declare void @foo(ptr byref(%struct) %a)

; CHECK: declare void @baz(ptr byref(%struct))
