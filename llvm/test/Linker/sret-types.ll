; RUN: llvm-link %s %p/Inputs/sret-type-input.ll -S | FileCheck %s

%a = type { i64 }
%struct = type { i32, i8 }

; CHECK-LABEL: define void @f(%a* sret(%a) %0)
define void @f(%a* sret(%a)) {
  ret void
}

; CHECK-LABEL: define void @bar(
; CHECK: call void @foo(%struct* sret(%struct) %ptr)
define void @bar() {
  %ptr = alloca %struct
  call void @foo(%struct* sret(%struct) %ptr)
  ret void
}

; CHECK-LABEL: define void @g(%a* sret(%a) %0)

; CHECK-LABEL: define void @foo(%struct* sret(%struct) %a)
; CHECK-NEXT:   call void @baz(%struct* sret(%struct) %a)
declare void @foo(%struct* sret(%struct) %a)

; CHECK: declare void @baz(%struct* sret(%struct))
