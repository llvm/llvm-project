; RUN: opt < %s -passes=print-callgraph -disable-output 2>&1 | FileCheck %s
; RUN: opt < %s -passes=print-callgraph -disable-output 2>&1 | FileCheck %s

; CHECK: Call graph node for function: 'caller'
; CHECK-NEXT:   CS<{{.*}}> calls function 'broker'
; CHECK-NEXT:   CS<None> calls function 'callback'

define void @caller(ptr %arg) {
  call void @broker(ptr @callback, ptr %arg)
  ret void
}

define void @callback(ptr %arg) {
  ret void
}

declare !callback !0 void @broker(ptr, ptr)

!0 = !{!1}
!1 = !{i64 0, i64 1, i1 false}
