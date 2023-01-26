; RUN: opt %s -passes=dot-callgraph -callgraph-multigraph -callgraph-dot-filename-prefix=%t -disable-output
; RUN: FileCheck %s -input-file=%t.callgraph.dot --check-prefix=CHECK-MULTIGRAPH
; RUN: opt %s -passes=dot-callgraph -callgraph-dot-filename-prefix=%t -disable-output
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
