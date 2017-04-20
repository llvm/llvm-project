; RUN: opt -S -structurizecfg %s | FileCheck %s

; CHECK-LABEL: @directly_invert_compare_condition_jump_into_loop(
; CHECK: %cmp0 = fcmp uge float %arg0, %arg1
; CHECK-NEXT: br i1 %cmp0, label %end.loop, label %Flow
define void @directly_invert_compare_condition_jump_into_loop(i32 addrspace(1)* %out, i32 %n, float %arg0, float %arg1) #0 {
entry:
  br label %for.body

for.body:
  %i = phi i32 [0, %entry], [%i.inc, %end.loop]
  %ptr = getelementptr i32, i32 addrspace(1)* %out, i32 %i
  store i32 %i, i32 addrspace(1)* %ptr, align 4
  %cmp0 = fcmp olt float %arg0, %arg1
  br i1 %cmp0, label %mid.loop, label %end.loop

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

; CHECK-LABEL: @invert_multi_use_compare_condition_jump_into_loop(
; CHECK: %cmp0 = fcmp olt float %arg0, %arg1
; CHECK: store volatile i1 %cmp0, i1 addrspace(1)* undef
; CHECK: %0 = xor i1 %cmp0, true
; CHECK-NEXT: br i1 %0, label %end.loop, label %Flow
define void @invert_multi_use_compare_condition_jump_into_loop(i32 addrspace(1)* %out, i32 %n, float %arg0, float %arg1) #0 {
entry:
  br label %for.body

for.body:
  %i = phi i32 [0, %entry], [%i.inc, %end.loop]
  %ptr = getelementptr i32, i32 addrspace(1)* %out, i32 %i
  store i32 %i, i32 addrspace(1)* %ptr, align 4
  %cmp0 = fcmp olt float %arg0, %arg1
  store volatile i1 %cmp0, i1 addrspace(1)* undef
  br i1 %cmp0, label %mid.loop, label %end.loop

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

attributes #0 = { nounwind }