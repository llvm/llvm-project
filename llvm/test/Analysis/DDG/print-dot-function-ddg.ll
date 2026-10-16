; The function-scoped DDG dot printer (-dot-function-ddg) is the whole-function
; counterpart to the loop-scoped -dot-ddg printer: it builds a
; DataDependenceGraph for the function and renders it with the shared DDG
; DOTGraphTraits. The output therefore matches -dot-ddg -- register def-use
; edges are blue and memory-dependence edges are red. -dot-function-ddg-only
; selects the simplified rendering (concise labels, synthetic root hidden).

; RUN: opt -aa-pipeline=basic-aa -passes=dot-function-ddg \
; RUN:     -dot-function-ddg-filename-prefix=%t < %s -disable-output
; RUN: FileCheck %s -input-file=%t.mem.dot

; RUN: opt -aa-pipeline=basic-aa -passes=dot-function-ddg -dot-function-ddg-only \
; RUN:     -dot-function-ddg-filename-prefix=%t.s < %s -disable-output
; RUN: FileCheck %s -input-file=%t.s.mem.dot -check-prefix=SIMPLE

;-----------------------------------------------------------------------------
; @mem: load -> add -> store to the same pointer. The store depends on the add
; (a register def-use) and on the load (a memory dependence). The verbose graph
; shows the synthetic root; the simplified graph hides it.
;-----------------------------------------------------------------------------
; CHECK:      digraph "DDG for 'mem'"
; The synthetic root and its rooted edges appear in the verbose graph:
; CHECK-DAG:  [[ROOT:Node0x[0-9a-f]+]] [shape=record,label="{\<kind:root\>\nroot\n}"]
; CHECK-DAG:  [[ROOT]] -> Node0x{{[0-9a-f]+}}[label="[rooted]"]
; Register def-use edges are blue:
; CHECK-DAG:  Node0x{{[0-9a-f]+}} -> Node0x{{[0-9a-f]+}}[label="[def-use]", color=blue]
; The load/store memory dependence is a red edge:
; CHECK-DAG:  Node0x{{[0-9a-f]+}} -> Node0x{{[0-9a-f]+}}[label="{{.*}}", color=red]
; Node labels carry the instruction text:
; CHECK-DAG:  label="{\<kind:single-instruction\>\n  %v = load i32, ptr %p, align 4\n}"
; CHECK-DAG:  label="{\<kind:single-instruction\>\n  %s = add i32 %v, %v\n}"
; CHECK-DAG:  label="{\<kind:single-instruction\>\n  store i32 %s, ptr %p, align 4\n}"

; Simplified: root hidden, concise labels (no "<kind:>" prefix), memory edge
; still present and labelled "[memory]".
; SIMPLE:      digraph "DDG for 'mem'"
; SIMPLE-DAG:  label="{  %v = load i32, ptr %p, align 4\n}"
; SIMPLE-DAG:  Node0x{{[0-9a-f]+}} -> Node0x{{[0-9a-f]+}}[label="[def-use]", color=blue]
; SIMPLE-DAG:  Node0x{{[0-9a-f]+}} -> Node0x{{[0-9a-f]+}}[label="[memory]", color=red]
; SIMPLE-NOT:  kind:
; SIMPLE-NOT:  [rooted]

define void @mem(ptr %p) {
  %v = load i32, ptr %p, align 4
  %s = add i32 %v, %v
  store i32 %s, ptr %p, align 4
  ret void
}
