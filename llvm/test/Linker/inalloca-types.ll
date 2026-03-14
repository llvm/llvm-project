; RUN: llvm-link %s %p/Inputs/inalloca-type-input.ll -S | FileCheck %s

%a = type { i64 }
%struct = type { i32, i8 }

; CHECK-LABEL: define void @f(ptr inalloca(%a) %0)
define void @f(ptr inalloca(%a)) {
  ret void
}

; CHECK-LABEL: define void @bar(
; CHECK: call void @foo(ptr inalloca(%struct) %ptr)
define void @bar() {
  %ptr = alloca inalloca %struct
  call void @foo(ptr inalloca(%struct) %ptr)
  ret void
}

; CHECK-LABEL: define void @g(ptr inalloca(%a) %0)

; CHECK-LABEL: define void @foo(ptr inalloca(%struct) %a)
; CHECK-NEXT:   call void @baz(ptr inalloca(%struct) %a)
declare void @foo(ptr inalloca(%struct) %a)

; CHECK: declare void @baz(ptr inalloca(%struct))
