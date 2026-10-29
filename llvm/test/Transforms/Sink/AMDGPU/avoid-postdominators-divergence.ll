; RUN: opt -S -mtriple=amdgcn -passes='sink<avoid-postdominators>' < %s | FileCheck %s

; On targets with divergent branches, only divergent values are counted.

; %a and %b are uniform, so keeping them live costs no vector registers.
define amdgpu_kernel void @sink_uniform_operands(i32 %a, i32 %b, i1 %c,
                                                 ptr addrspace(1) %p) {
; CHECK-LABEL: @sink_uniform_operands(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br i1 %c, label %then, label %merge
; CHECK:       merge:
; CHECK-NEXT:    %m = mul i32 %a, %b
;
entry:
  %m = mul i32 %a, %b
  br i1 %c, label %then, label %merge

then:
  store i32 0, ptr addrspace(1) %p
  br label %merge

merge:
  store i32 %m, ptr addrspace(1) %p
  ret void
}

; %a and %b are divergent.
define amdgpu_kernel void @no_sink_divergent_operands(ptr addrspace(1) %p,
                                                      i1 %c) {
; CHECK-LABEL: @no_sink_divergent_operands(
; CHECK:         %m = mul i32 %a, %b
; CHECK-NEXT:    br i1 %c, label %then, label %merge
; CHECK:       merge:
; CHECK-NEXT:    store i32 %m
;
entry:
  %a = call i32 @llvm.amdgcn.workitem.id.x()
  %b = call i32 @llvm.amdgcn.workitem.id.y()
  %m = mul i32 %a, %b
  br i1 %c, label %then, label %merge

then:
  store i32 0, ptr addrspace(1) %p
  br label %merge

merge:
  store i32 %m, ptr addrspace(1) %p
  ret void
}

; %i and %i.next are uniform at their definitions, but the loop has a
; divergent exit, so they are divergent when used after the loop.
define amdgpu_kernel void
@no_sink_temporally_divergent_operands(ptr addrspace(1) %p, i1 %c) {
; CHECK-LABEL: @no_sink_temporally_divergent_operands(
; CHECK:       after:
; CHECK-NEXT:    %m = mul i32 %i.next, %i
; CHECK-NEXT:    br i1 %c, label %then, label %merge
; CHECK:       merge:
; CHECK-NEXT:    store i32 %m
;
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  br label %loop

loop:
  %i = phi i32 [ 0, %entry ], [ %i.next, %loop ]
  %i.next = add i32 %i, 1
  %done = icmp eq i32 %i.next, %tid
  br i1 %done, label %after, label %loop

after:
  %m = mul i32 %i.next, %i
  br i1 %c, label %then, label %merge

then:
  store i32 0, ptr addrspace(1) %p
  br label %merge

merge:
  store i32 %m, ptr addrspace(1) %p
  ret void
}
