; RUN: opt -S -o - -structurizecfg < %s | FileCheck %s

; CHECK-LABEL: @invert_branch_on_arg_inf_loop(
; CHECK: entry:
; CHECK: %arg.inv = xor i1 %arg, true
; CHECK: phi i1 [ false, %Flow1 ], [ %arg.inv, %entry ]
define void @invert_branch_on_arg_inf_loop(i32 addrspace(1)* %out, i1 %arg) {
entry:
  br i1 %arg, label %for.end, label %for.body

for.body:                                         ; preds = %entry, %for.body
  store i32 999, i32 addrspace(1)* %out, align 4
  br label %for.body

for.end:                                          ; preds = %Flow
  ret void
}


; CHECK-LABEL: @invert_branch_on_arg_jump_into_loop(
; CHECK: entry:
; CHECK: %arg.inv = xor i1 %arg, true
; CHECK: Flow:
; CHECK: Flow1:
define void @invert_branch_on_arg_jump_into_loop(i32 addrspace(1)* %out, i32 %n, i1 %arg) {
entry:
  br label %for.body

for.body:
  %i = phi i32 [0, %entry], [%i.inc, %end.loop]
  %ptr = getelementptr i32 addrspace(1)* %out, i32 %i
  store i32 %i, i32 addrspace(1)* %ptr, align 4
  br i1 %arg, label %mid.loop, label %end.loop

mid.loop:
  store i32 333, i32 addrspace(1)* %out, align 4
  br label %for.end

end.loop:
  %i.inc = add i32 %i, 1
  %cmp = icmp ne i32 %i.inc, %n
  br i1 %cmp, label %for.body, label %for.end

for.end:
  ret void
}

