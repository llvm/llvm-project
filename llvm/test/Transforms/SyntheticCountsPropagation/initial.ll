; RUN: opt -passes=synthetic-counts-propagation -S < %s | FileCheck %s

; CHECK-LABEL: define void @foo()
; CHECK: !prof ![[COUNT1:[0-9]+]]
define void @foo() {
  ret void
}

; CHECK-LABEL: define void @foo_inline() #0
; CHECK: !prof ![[COUNT2:[0-9]+]]
define void @foo_inline() #0 {
  ret void
}

; CHECK-LABEL: define void @foo_always_inline() #1
; CHECK: !prof ![[COUNT2]]
define void @foo_always_inline() #1 {
  ret void
}

; CHECK-LABEL: define void @foo_cold() #2
; CHECK: !prof ![[COUNT3:[0-9]+]]
define void @foo_cold() #2 {
  ret void
}

; CHECK-LABEL: define void @foo_noinline() #3
; CHECK: !prof ![[COUNT3]]
define void @foo_noinline() #3 {
  ret void
}

; CHECK-LABEL: define internal void @foo_local()
; CHECK: !prof ![[COUNT4:[0-9]+]]
define internal void @foo_local() {
  ret void
}

; CHECK-LABEL: define internal void @foo_local_escaped()
; CHECK: !prof ![[COUNT1]]
define internal void @foo_local_escaped() {
  ret void
}

declare void @ext(ptr)

define void @bar() {
  call void @ext(ptr nonnull @foo_local_escaped)
  ret void
}

; CHECK-LABEL: define internal void @foo_local_inline() #0
; CHECK: !prof ![[COUNT2]]
define internal void @foo_local_inline() #0 {
  ret void
}

; CHECK-LABEL: define internal void @foo_local_cold() #2
; CHECK: !prof ![[COUNT4]]
define internal void @foo_local_cold() #2 {
  ret void
}

; CHECK-LABEL: define linkonce void @foo_linkonce()
; CHECK: !prof ![[COUNT1]]
define linkonce void @foo_linkonce() {
  ret void
}

; CHECK: ![[COUNT1]] = !{!"synthetic_function_entry_count", i64 10}
; CHECK: ![[COUNT2]] = !{!"synthetic_function_entry_count", i64 15}
; CHECK: ![[COUNT3]] = !{!"synthetic_function_entry_count", i64 5}
; CHECK: ![[COUNT4]] = !{!"synthetic_function_entry_count", i64 0}

attributes #0 = {inlinehint}
attributes #1 = {alwaysinline}
attributes #2 = {cold}
attributes #3 = {noinline}

