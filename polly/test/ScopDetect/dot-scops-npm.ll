; RUN: opt %loadPolly "-passes=polly-scop-printer" -disable-output < %s
; RUN: FileCheck %s -input-file=scops.func.dot
;
; Check that the ScopPrinter does not crash.
; ScopPrinter needs the ScopDetection pass, which should depend on
; ScalarEvolution transitively.

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @func(i32 %n, i32 %m, double* noalias nonnull %A) {
; CHECK:      digraph "Scop Graph for 'func' function"
; CHECK-NEXT: label="Scop Graph for 'func' function"
; CHECK:      Node0x[[EntryID:.*]] [shape=record,label="{entry:\l  br label %outer.for\l}"];
; CHECK-NEXT: Node0x[[EntryID]] -> Node0x[[OUTER_FOR_ID:.*]];
; CHECK-NEXT: Node0x[[OUTER_FOR_ID]] [shape=record,label="{outer.for:
; CHECK-NEXT: Node0x[[OUTER_FOR_ID]] -> Node0x[[INNER_FOR_ID:.*]];
; CHECK-NEXT: Node0x[[OUTER_FOR_ID]] -> Node0x[[OUTER_EXIT:.*]];
; CHECK-NEXT: Node0x[[INNER_FOR_ID]] [shape=record,label="{inner.for:
; CHECK-NEXT: Node0x[[INNER_FOR_ID]] -> Node0x[[BABY1_ID:.*]];
; CHECK-NEXT: Node0x[[INNER_FOR_ID]] -> Node0x[[INNER_EXIT_ID:.*]];
; CHECK-NEXT: Node0x[[BABY1_ID]] [shape=record,label="{body1:
; CHECK-NEXT: Node0x[[BABY1_ID]] -> Node0x[[INNER_INC_ID:.*]];
; CHECK-NEXT: Node0x[[INNER_INC_ID]] [shape=record,label="{inner.inc:
; CHECK-NEXT: Node0x[[INNER_INC_ID]] -> Node0x[[INNER_FOR_ID]][constraint=false];
; CHECK-NEXT: Node0x[[INNER_EXIT_ID]] [shape=record,label="{inner.exit:
; CHECK-NEXT: Node0x[[INNER_EXIT_ID]] -> Node0x[[OUTER_INC_ID:.*]];
; CHECK-NEXT: Node0x[[OUTER_INC_ID]] [shape=record,label="{outer.inc:
; CHECK-NEXT: Node0x[[OUTER_INC_ID]] -> Node0x[[OUTER_FOR_ID]][constraint=false];
; CHECK-NEXT: Node0x[[OUTER_EXIT]] [shape=record,label="{outer.exit:
; CHECK-NEXT: Node0x[[OUTER_EXIT]] -> Node0x[[RETURN_ID:.*]];
; CHECK-NEXT: Node0x[[RETURN_ID]] [shape=record,label="{return:
; CHECK-NEXT: colorscheme = "paired12"
; CHECK_NEXT: subgraph cluster_0x[[:.*]] {
; CHECK_NEXT: label = "";
; CHECK_NEXT: style = solid;
; CHECK_NEXT: color = 1
; CHECK_NEXT: subgraph cluster_0x[[:.*]] {
; CHECK_NEXT: label = "";
; CHECK_NEXT: style = filled;
; CHECK_NEXT: color = 3            subgraph cluster_0x7152c40 {
; CHECK_NEXT: label = "";
; CHECK_NEXT: style = solid;
; CHECK_NEXT: color = 5
; CHECK_NEXT: subgraph cluster_0x[[:.*]] {
; CHECK_NEXT: label = "";
; CHECK_NEXT: style = solid;
; CHECK_NEXT: color = 7
; CHECK_NEXT: Node0x[[INNER_FOR_ID]];
; CHECK_NEXT: Node0x[[BABY1_ID]];
; CHECK_NEXT: Node0x[[INNER_INC_ID]];
; CHECK_NEXT: }
; CHECK_NEXT: Node0x[[OUTER_FOR_ID]];
; CHECK_NEXT: Node0x[[INNER_EXIT_ID]];
; CHECK_NEXT: Node0x[[OUTER_INC_ID]];
; CHECK_NEXT: }
; CHECK_NEXT: Node0x[[OUTER_EXIT]];
; CHECK_NEXT: }
; CHECK_NEXT: Node0x[[EntryID]];
; CHECK_NEXT: Node0x[[RETURN_ID]];
; CHECK_NEXT: }
; CHECK_NEXT: }

entry:
  br label %outer.for

outer.for:
  %j = phi i32 [0, %entry], [%j.inc, %outer.inc]
  %j.cmp = icmp slt i32 %j, %n
  br i1 %j.cmp, label %inner.for, label %outer.exit

  inner.for:
    %i = phi i32 [1, %outer.for], [%i.inc, %inner.inc]
    %b = phi double [0.0, %outer.for], [%a, %inner.inc]
    %i.cmp = icmp slt i32 %i, %m
    br i1 %i.cmp, label %body1, label %inner.exit

    body1:
      %A_idx = getelementptr inbounds double, double* %A, i32 %i
      %a = load double, double* %A_idx
      store double %a, double* %A_idx
      br label %inner.inc

  inner.inc:
    %i.inc = add nuw nsw i32 %i, 1
    br label %inner.for

  inner.exit:
    br label %outer.inc

outer.inc:
  store double %b, double* %A
  %j.inc = add nuw nsw i32 %j, 1
  br label %outer.for

outer.exit:
  br label %return

return:
  ret void
}
