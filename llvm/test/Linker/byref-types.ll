; RUN: llvm-link %s %p/Inputs/byref-type-input.ll -S | FileCheck %s

%a = type { i64 }
%struct = type { i32, i8 }

; CHECK-LABEL: define void @f(%a* byref(%a) %0)
define void @f(%a* byref(%a)) {
  ret void
}

; CHECK-LABEL: define void @bar(
; CHECK: call void @foo(%struct* byref(%struct) %ptr)
define void @bar() {
  %ptr = alloca %struct
  call void @foo(%struct* byref(%struct) %ptr)
  ret void
}

; CHECK-LABEL: define void @g(%a* byref(%a) %0)

; CHECK-LABEL: define void @foo(%struct* byref(%struct) %a)
; CHECK-NEXT:   call void @baz(%struct* byref(%struct) %a)
declare void @foo(%struct* byref(%struct) %a)

; CHECK: declare void @baz(%struct* byref(%struct))
