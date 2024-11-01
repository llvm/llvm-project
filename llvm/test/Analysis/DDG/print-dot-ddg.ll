; RUN: opt -aa-pipeline=basic-aa -passes=dot-ddg -dot-ddg-filename-prefix=%t < %s 2>&1 > /dev/null
; RUN: FileCheck %s -input-file=%t.foo.for.body.dot
; RUN: opt -aa-pipeline=basic-aa -passes=dot-ddg -dot-ddg-filename-prefix=%t -dot-ddg-only < %s 2>&1 > /dev/null
; RUN: FileCheck %s -input-file=%t.foo.for.body.dot -check-prefix=CHECK-ONLY

target datalayout = "e-m:e-i64:64-n32:64-v256:256:256-v512:512:512"

; Test the dot graph printer for a non-trivial DDG graph generated from
; the following test case. In particular it tests that pi-blocks are
; printed properly and that multiple memory dependencies on a single edge
; are shown in the full dot graph.
;
; void foo(ptr restrict A, ptr restrict B, int n) {
;   for (int i = 0; i < n; i++) {
;     A[i] = A[i] + B[i];
;     B[i+1] = A[i] + 1;
;   }
; }


; CHECK: digraph "DDG for 'foo.for.body'"
; CHECK-NEXT: label="DDG for 'foo.for.body'";
; CHECK: {{Node0x.*}} [shape=record,label="{\<kind:root\>\nroot\n}"]
; CHECK: {{Node0x.*}} -> {{Node0x.*}}[label="[rooted]"]
; CHECK-COUNT-6: {{Node0x.*}} -> {{Node0x.*}}[label="[def-use]"]
; CHECK-NOT: {{Node0x.*}} -> {{Node0x.*}}[label="[def-use]"]
; CHECK: [shape=record,label="{\<kind:single-instruction\>\n  %arrayidx10 = getelementptr inbounds float, ptr %B, i64 %indvars.iv.next\n}"];
; CHECK: [shape=record,label="{\<kind:multi-instruction\>\n  %arrayidx = getelementptr inbounds float, ptr %A, i64 %indvars.iv\n  %0 = load float, ptr %arrayidx, align 4\n}"];
; CHECK: {{Node0x.*}} -> {{Node0x.*}}[label="[consistent anti [0|<]!, consistent input [0|<]!]"]
; CHECK: [shape=record,label="{\<kind:pi-block\>\n--- start of nodes in pi-block ---\n\<kind:single-instruction\>\n  %1 = load float, ptr %arrayidx2, align 4\n\n\<kind:single-instruction\>\n  %add = fadd fast float %0, %1\n\n\<kind:single-instruction\>\n  store float %add, ptr %arrayidx4, align 4\n\n\<kind:multi-instruction\>\n  %2 = load float, ptr %arrayidx6, align 4\n  %add7 = fadd fast float %2, 1.000000e+00\n\n\<kind:single-instruction\>\n  store float %add7, ptr %arrayidx10, align 4\n--- end of nodes in pi-block ---\n}"];

; CHECK-ONLY: digraph "DDG for 'foo.for.body'"
; CHECK-ONLY-NEXT: label="DDG for 'foo.for.body'";
; CHECK-ONLY: [shape=record,label="{pi-block\nwith\n2 nodes\n}"];
; CHECK-ONLY-COUNT-6: {{Node0x.*}} -> {{Node0x.*}}[label="[def-use]"];
; CHECK-NOT: {{Node0x.*}} -> {{Node0x.*}}[label="[def-use]"];
; CHECK-ONLY: [shape=record,label="{  %arrayidx10 = getelementptr inbounds float, ptr %B, i64 %indvars.iv.next\n}"];
; CHECK-ONLY: [shape=record,label="{  %arrayidx = getelementptr inbounds float, ptr %A, i64 %indvars.iv\n  %0 = load float, ptr %arrayidx, align 4\n}"];
; CHECK-ONLY: {{Node0x.*}} -> {{Node0x.*}}[label="[memory]"]
; CHECK-ONLY: [shape=record,label="{pi-block\nwith\n5 nodes\n}"];

define void @foo(ptr noalias %A, ptr noalias %B, i32 signext %n) {
entry:
  %cmp1 = icmp sgt i32 %n, 0
  br i1 %cmp1, label %for.body.preheader, label %for.end

for.body.preheader:                               ; preds = %entry
  %wide.trip.count = zext i32 %n to i64
  br label %for.body

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds float, ptr %A, i64 %indvars.iv
  %0 = load float, ptr %arrayidx, align 4
  %arrayidx2 = getelementptr inbounds float, ptr %B, i64 %indvars.iv
  %1 = load float, ptr %arrayidx2, align 4
  %add = fadd fast float %0, %1
  %arrayidx4 = getelementptr inbounds float, ptr %A, i64 %indvars.iv
  store float %add, ptr %arrayidx4, align 4
  %arrayidx6 = getelementptr inbounds float, ptr %A, i64 %indvars.iv
  %2 = load float, ptr %arrayidx6, align 4
  %add7 = fadd fast float %2, 1.000000e+00
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %arrayidx10 = getelementptr inbounds float, ptr %B, i64 %indvars.iv.next
  store float %add7, ptr %arrayidx10, align 4
  %exitcond = icmp ne i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond, label %for.body, label %for.end.loopexit

for.end.loopexit:                                 ; preds = %for.body
  br label %for.end

for.end:                                          ; preds = %for.end.loopexit, %entry
  ret void
}
