; RUN: llvm-link %s %p/Inputs/byval-types-1.ll -S | FileCheck %s

%struct = type {i32, i8}

declare void @foo(ptr byval(%struct) %a)

define void @bar() {
  %ptr = alloca %struct
; CHECK: call void @foo(ptr byval(%struct) %ptr)
  call void @foo(ptr byval(%struct) %ptr)
  ret void
}

; CHECK: define void @foo(ptr byval(%struct) %a)
; CHECK-NEXT:   call void @baz(ptr byval(%struct) %a)

; CHECK: declare void @baz(ptr byval(%struct))
