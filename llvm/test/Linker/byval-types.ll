; RUN: llvm-link %s %p/Inputs/byval-types-1.ll -S | FileCheck %s

%struct = type {i32, i8}

declare void @foo(%struct* byval(%struct) %a)

define void @bar() {
  %ptr = alloca %struct
; CHECK: call void @foo(%struct* byval(%struct) %ptr)
  call void @foo(%struct* byval(%struct) %ptr)
  ret void
}

; CHECK: define void @foo(%struct* byval(%struct) %a)
; CHECK-NEXT:   call void @baz(%struct* byval(%struct) %a)

; CHECK: declare void @baz(%struct* byval(%struct))
