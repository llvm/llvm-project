; RUN: llc -mtriple=hexagon -O2 < %s | FileCheck %s
;
; A pointer-chasing loop whose trip count is not computable: the traversal
; pointer is reloaded from memory each iteration (n = n->next) and the loop
; exits when it becomes null.  The post-increment load in this loop defines
; two registers -- the loaded value and the incremented address -- and only
; the incremented address is a "base + constant" induction variable.
;
; HexagonHardwareLoops must not mistake the loaded value for the bumped
; address; doing so makes it derive a trip count from the numeric value of
; the initial head pointer (sub(#0,rN) / lsr(rN,#2)) and drop the null test
; from the loop body, so the loop runs past the end of the list.

; CHECK-LABEL: find:
; CHECK-NOT: loop0
; CHECK-NOT: endloop0

%struct.S = type { i32, i32 }
%struct.N = type { ptr, %struct.S }

define ptr @find(ptr %head, i32 %t) {
entry:
  %cmp.not6 = icmp eq ptr %head, null
  br i1 %cmp.not6, label %for.end, label %for.body

for.body:
  %r.08 = phi ptr [ %spec.select, %for.body ], [ null, %entry ]
  %n.07 = phi ptr [ %next, %for.body ], [ %head, %entry ]
  %s = getelementptr inbounds nuw i8, ptr %n.07, i32 4
  %valp = getelementptr inbounds nuw i8, ptr %n.07, i32 8
  %val = load i32, ptr %valp, align 4
  %cmp1 = icmp eq i32 %val, %t
  %next = load ptr, ptr %n.07, align 4
  %spec.select = select i1 %cmp1, ptr %s, ptr %r.08
  %cmp.not = icmp eq ptr %next, null
  br i1 %cmp.not, label %for.end, label %for.body

for.end:
  %r.0.lcssa = phi ptr [ null, %entry ], [ %spec.select, %for.body ]
  ret ptr %r.0.lcssa
}
