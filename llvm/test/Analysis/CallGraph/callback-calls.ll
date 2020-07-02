; RUN: opt < %s -print-callgraph -disable-output 2>&1 | FileCheck %s
; RUN: opt < %s -passes=print-callgraph -disable-output 2>&1 | FileCheck %s

; CHECK: Call graph node for function: 'caller'
; CHECK-NEXT:   CS<{{.*}}> calls function 'broker'
; CHECK-NEXT:   CS<None> calls function 'callback'

define void @caller(i32* %arg) {
  call void @broker(void (i32*)* @callback, i32* %arg)
  ret void
}

define void @callback(i32* %arg) {
  ret void
}

declare !callback !0 void @broker(void (i32*)*, i32*)

!0 = !{!1}
!1 = !{i64 0, i64 1, i1 false}
