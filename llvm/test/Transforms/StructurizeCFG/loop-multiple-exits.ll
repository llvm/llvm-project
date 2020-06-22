; RUN: opt -S -structurizecfg %s -o - | FileCheck %s
;
; void loop(int *out, int cond_a, int cond_b) {
;
;   unsigned i;
;   for (i = 0; i < cond_a; i++) {
;     out[i] = i;
;     if (i > cond_b) {
;       break;
;     }
;     out[i + cond_a] = i;
;   }
; }

define void @loop(i32 addrspace(1)* %out, i32 %cond_a, i32 %cond_b) nounwind uwtable {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %i.0 = phi i32 [ 0, %entry ], [ %inc, %for.inc ]
  %cmp = icmp ult i32 %i.0, %cond_a
  br i1 %cmp, label %for.body, label %for.end

; CHECK: for.body:
for.body:                                         ; preds = %for.cond
  %arrayidx = getelementptr inbounds i32, i32 addrspace(1)* %out, i32 %i.0
  store i32 %i.0, i32 addrspace(1)* %arrayidx, align 4
  %cmp1 = icmp ugt i32 %i.0, %cond_b
; CHECK: br i1 %{{[0-9a-zA-Z_.]+}}, label %for.inc, label %[[FLOW1:[0-9a-zA-Z_]+]]
  br i1 %cmp1, label %for.end, label %for.inc

; CHECK: [[FLOW:[0-9a-zA-Z]+]]:
; CHECK: br i1 %{{[0-9a-zA-Z_.]+}}, label %for.end, label %for.cond

; CHECK: for.inc:
; CHECK: br label %[[FLOW1]]

for.inc:                                          ; preds = %for.body
  %0 = add i32 %cond_a, %i.0
  %arrayidx3 = getelementptr inbounds i32, i32 addrspace(1)* %out, i32 %0
  store i32 %i.0, i32 addrspace(1)* %arrayidx3, align 4
  %inc = add i32 %i.0, 1
  br label %for.cond

; CHECK: [[FLOW1]]
; CHECK: br label %[[FLOW]]

for.end:                                          ; preds = %for.cond, %for.body
  ret void
}
