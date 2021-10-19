; RUN: llvm-link %s %p/Inputs/inalloca-type-input.ll -S | FileCheck %s

%a = type { i64 }
%struct = type { i32, i8 }

; CHECK-LABEL: define void @f(%a* inalloca(%a) %0)
define void @f(%a* inalloca(%a)) {
  ret void
}

; CHECK-LABEL: define void @bar(
; CHECK: call void @foo(%struct* inalloca(%struct) %ptr)
define void @bar() {
  %ptr = alloca inalloca %struct
  call void @foo(%struct* inalloca(%struct) %ptr)
  ret void
}

; CHECK-LABEL: define void @g(%a* inalloca(%a) %0)

; CHECK-LABEL: define void @foo(%struct* inalloca(%struct) %a)
; CHECK-NEXT:   call void @baz(%struct* inalloca(%struct) %a)
declare void @foo(%struct* inalloca(%struct) %a)

; CHECK: declare void @baz(%struct* inalloca(%struct))
