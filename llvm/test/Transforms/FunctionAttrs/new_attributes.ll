; RUN: opt < %s -attributor -attributor-annotate-decl-cs -attributor-disable=false -attributor-max-iterations=0 -S | FileCheck %s
; RUN: opt < %s -attributor -attributor-annotate-decl-cs -attributor-disable=false -attributor-max-iterations=1 -S | FileCheck %s
; RUN: opt < %s -attributor -attributor-annotate-decl-cs -attributor-disable=false -attributor-max-iterations=2 -S | FileCheck %s
; RUN: opt < %s -attributor -attributor-annotate-decl-cs -attributor-disable=false -attributor-max-iterations=3 -S | FileCheck %s
; RUN: opt < %s -attributor -attributor-annotate-decl-cs -attributor-disable=false -attributor-max-iterations=4 -S | FileCheck %s
; RUN: opt < %s -attributor -attributor-annotate-decl-cs -attributor-disable=false -attributor-max-iterations=2147483647 -S | FileCheck %s

; CHECK-NOT: Function
; CHECK: declare i32 @foo1()
; CHECK-NOT: Function
; CHECK: declare i32 @foo2()
; CHECK-NOT: Function
; CHECK: declare i32 @foo3()
declare i32 @foo1()
declare i32 @foo2()
declare i32 @foo3()

; CHECK-NOT: Function
; CHECK:      define internal i32 @bar() {
; CHECK-NEXT:   %1 = call i32 @foo1()
; CHECK-NEXT:   %2 = call i32 @foo2()
; CHECK-NEXT:   %3 = call i32 @foo3()
; CHECK-NEXT:   ret i32 undef
; CHECK-NEXT: }
define internal i32 @bar() {
  %1 = call i32 @foo1()
  %2 = call i32 @foo2()
  %3 = call i32 @foo3()
  ret i32 1
}

; CHECK-NOT: Function
; CHECK:      define i32 @baz() {
; CHECK-NEXT:   %1 = call i32 @bar()
; CHECK-NEXT:   ret i32 0
; CHECK-NEXT: }
define i32 @baz() {
  %1 = call i32 @bar()
  ret i32 0
}

; We should never derive anything here
; CHECK-NOT: attributes
