; RUN: opt %s -dot-callgraph -callgraph-multigraph -callgraph-dot-filename-prefix=%t 2>/dev/null
; RUN: FileCheck %s -input-file=%t.callgraph.dot --check-prefix=CHECK-MULTIGRAPH
; RUN: opt %s -dot-callgraph -callgraph-dot-filename-prefix=%t 2>/dev/null
; RUN: FileCheck %s -input-file=%t.callgraph.dot --check-prefix=CHECK

; CHECK-MULTIGRAPH: {external caller}
; CHECK-NOT: {external caller}

define void @bar() {
  ret void
}

define void @foo() {
  call void @bar()
  ret void
}
