; RUN: llvm-link %s %p/Inputs/sret-type-input.ll -S | FileCheck %s

%a = type { i64 }
%struct = type { i32, i8 }

; CHECK-LABEL: define void @f(ptr sret(%a) %0)
define void @f(ptr sret(%a)) {
  ret void
}

; CHECK-LABEL: define void @bar(
; CHECK: call void @foo(ptr sret(%struct) %ptr)
define void @bar() {
  %ptr = alloca %struct
  call void @foo(ptr sret(%struct) %ptr)
  ret void
}

; CHECK-LABEL: define void @g(ptr sret(%a) %0)

; CHECK-LABEL: define void @foo(ptr sret(%struct) %a)
; CHECK-NEXT:   call void @baz(ptr sret(%struct) %a)
declare void @foo(ptr sret(%struct) %a)

; CHECK: declare void @baz(ptr sret(%struct))
