; RUN: opt < %s -passes=argpromotion -S | FileCheck %s

; CHECK-LABEL: define i32 @foo() #0 {
; CHECK-NEXT:      %.val = load <32 x half>, ptr poison, align 4
; CHECK-NEXT:      call void @bar(<32 x half> %.val)
; CHECK-NEXT:      ret i32 0
; CHECK-NEXT:    }

; CHECK-LABEL: define internal void @bar(<32 x half> %.0.val) #0 {
; CHECK-NEXT:      ret void
; CHECK-NEXT:    }

; CHECK:    attributes #0 = { uwtable "min-legal-vector-width"="512" }

define i32 @foo() #0 {
  call void @bar(ptr poison)
  ret i32 0
}

define internal void @bar(ptr) #0 {
  %2 = load <32 x half>, ptr %0, align 4
  ret void
}

attributes #0 = { uwtable "min-legal-vector-width"="0" }
