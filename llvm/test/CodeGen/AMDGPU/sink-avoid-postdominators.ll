; RUN: llc -mtriple=amdgcn -mcpu=gfx1200 -stop-after=sink < %s | FileCheck %s

; AMDGPU runs the pre-ISel sink with avoid-postdominators. %m is only used
; after the diamond, so sinking it into %merge would keep %a and %b live across
; the diamond without saving any execution.

; CHECK-LABEL: @keep_out_of_postdominator(
; CHECK:       entry:
; CHECK:         %m = fmul <4 x float> %a, %b
; CHECK:       merge:
; CHECK-NOT:     fmul
; CHECK:         store <4 x float> %m
define amdgpu_kernel void @keep_out_of_postdominator(ptr addrspace(1) %in,
                                                     ptr addrspace(1) %out,
                                                     i32 %n) {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %pa = getelementptr <4 x float>, ptr addrspace(1) %in, i32 %tid
  %a = load <4 x float>, ptr addrspace(1) %pa
  %pb = getelementptr <4 x float>, ptr addrspace(1) %pa, i32 64
  %b = load <4 x float>, ptr addrspace(1) %pb
  %m = fmul <4 x float> %a, %b
  %c = icmp ult i32 %tid, %n
  br i1 %c, label %then, label %merge

then:
  %pt = getelementptr i32, ptr addrspace(1) %out, i32 %tid
  store i32 0, ptr addrspace(1) %pt
  br label %merge

merge:
  %po = getelementptr <4 x float>, ptr addrspace(1) %out, i32 %tid
  store <4 x float> %m, ptr addrspace(1) %po
  ret void
}

; Uniform values live in SGPRs and do not affect VGPR pressure, so sinking the
; extract into the postdominating block is still allowed.

; CHECK-LABEL: @sink_uniform_into_postdominator(
; CHECK:       entry:
; CHECK-NOT:     extractelement
; CHECK:       merge:
; CHECK:         %e = extractelement <5 x i32> %v{{.*}}, i64 3
define amdgpu_kernel void @sink_uniform_into_postdominator(
    <5 x i32> %v, ptr addrspace(1) %out, i32 %n) {
entry:
  %e = extractelement <5 x i32> %v, i64 3
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %c = icmp ult i32 %tid, %n
  br i1 %c, label %then, label %merge

then:
  %pt = getelementptr i32, ptr addrspace(1) %out, i32 %tid
  store i32 0, ptr addrspace(1) %pt
  br label %merge

merge:
  store i32 %e, ptr addrspace(1) %out
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x()
