; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK: %[[#Var:]] = OpPhi
; CHECK: OpSwitch %[[#Var]] %[[#]] [[#]] %[[#]] [[#]] %[[#]] [[#]] %[[#]] [[#]] %[[#]] [[#]] %[[#]] [[#]] %[[#]] [[#]] %[[#]] [[#]] %[[#]] [[#]] %[[#]] [[#]] %[[#]] [[#]] %[[#]] [[#]] %[[#]]
; CHECK-COUNT-11: OpBranch
; CHECK-NOT: OpBranch

define spir_func void @foo(i64 noundef %addr, i64 noundef %as) {
entry:
  %0 = inttoptr i64 %as to ptr addrspace(4)
  %1 = load i8, ptr addrspace(4) %0
  %cmp = icmp sgt i8 %1, 0
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  %add.ptr = getelementptr inbounds i8, ptr addrspace(4) %0, i64 1
  %2 = load i8, ptr addrspace(4) %add.ptr
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  %shadow_value.0.in = phi i8 [ %2, %if.then ], [ %1, %entry ]
  switch i8 %shadow_value.0.in, label %sw.default [
    i8 -127, label %sw.epilog
    i8 -126, label %sw.bb3
    i8 -125, label %sw.bb4
    i8 -111, label %sw.bb5
    i8 -110, label %sw.bb6
    i8 -109, label %sw.bb7
    i8 -15, label %sw.bb8
    i8 -14, label %sw.bb8
    i8 -13, label %sw.bb8
    i8 -124, label %sw.bb9
    i8 -95, label %sw.bb10
    i8 -123, label %sw.bb11
  ]

sw.bb3:                                           ; preds = %if.end
  br label %sw.epilog

sw.bb4:                                           ; preds = %if.end
  br label %sw.epilog

sw.bb5:                                           ; preds = %if.end
  br label %sw.epilog

sw.bb6:                                           ; preds = %if.end
  br label %sw.epilog

sw.bb7:                                           ; preds = %if.end
  br label %sw.epilog

sw.bb8:                                           ; preds = %if.end, %if.end, %if.end
  br label %sw.epilog

sw.bb9:                                           ; preds = %if.end
  br label %sw.epilog

sw.bb10:                                          ; preds = %if.end
  br label %sw.epilog

sw.bb11:                                          ; preds = %if.end
  br label %sw.epilog

sw.default:                                       ; preds = %if.end
  br label %sw.epilog

sw.epilog:                                        ; preds = %sw.default, %sw.bb11, %sw.bb10, %sw.bb9, %sw.bb8, %sw.bb7, %sw.bb6, %sw.bb5, %sw.bb4, %sw.bb3, %if.end
  br label %exit

if.then.i:                                        ; preds = %sw.epilog
  br label %exit

for.cond.i:                                       ; preds = %for.inc.i, %if.then.i
  br label %exit

for.inc.i:                                        ; preds = %for.cond.i
  br label %exit

if.end.i:                                         ; preds = %for.cond.i, %if.then.i
  br label %exit

if.end18.thread.i:                                ; preds = %if.end.i
  br label %5

for.cond8.i:                                      ; preds = %for.inc14.i, %if.end.i
  br label %exit

for.inc14.i:                                      ; preds = %for.cond8.i
  br label %exit

if.end18.i:                                       ; preds = %for.cond8.i
  br label %5

5:                                                ; preds = %if.end18.i, %if.end18.thread.i
  br label %for.cond25.i

for.cond25.i:                                     ; preds = %for.body29.i, %5
  br label %exit

for.cond.cleanup27.i:                             ; preds = %for.cond25.i
  br label %for.cond41.i

for.body29.i:                                     ; preds = %for.cond25.i
  br label %for.cond25.i

for.cond41.i:                                     ; preds = %for.body45.i, %for.cond.cleanup27.i
  br label %exit

for.cond.cleanup43.i:                             ; preds = %for.cond41.i
  br label %exit

for.body45.i:                                     ; preds = %for.cond41.i
  br label %for.cond41.i

exit:
  ret void
}
