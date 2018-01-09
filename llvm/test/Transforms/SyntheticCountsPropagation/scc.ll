; RUN: opt -passes=synthetic-counts-propagation -S < %s | FileCheck %s

; CHECK-LABEL: define void @foo()
; CHECK: !prof ![[COUNT1:[0-9]+]]
define void @foo() {
  call void @bar()
  ret void
}

; CHECK-LABEL: define void @bar() #0
; CHECK: !prof ![[COUNT1]]
define void @bar() #0 {
  call void @foo()
  ret void
}

attributes #0 = {inlinehint}

; CHECK: ![[COUNT1]] = !{!"synthetic_function_entry_count", i64 25}
