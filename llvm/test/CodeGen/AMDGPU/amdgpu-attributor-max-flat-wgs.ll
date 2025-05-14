; RUN: opt -S -mtriple=amdgcn-unknown-amdhsa -mcpu=gfx942 -passes=amdgpu-attributor %s | FileCheck %s

; CHECK-LABEL: define internal fastcc void @call1(
; CHECK-SAME: ) #[[ATTR0:[0-9]+]]
define internal fastcc void @call1() #0 {
  tail call fastcc void @call2()
  ret void
}

; CHECK-LABEL: define internal fastcc void @call2(
; CHECK-SAME: ) #[[ATTR0]]
define internal fastcc void @call2() #1 {
  tail call fastcc void @call5()
  ret void
}

; CHECK-LABEL: define { ptr addrspace(1), ptr } @call3(
; CHECK-SAME:) #[[ATTR0]]
define { ptr addrspace(1), ptr } @call3() #2 {
  tail call fastcc void @call5()
  ret { ptr addrspace(1), ptr } zeroinitializer
}

; CHECK-LABEL: define internal fastcc void @call5(
; CHECK-SAME: ) #[[ATTR0]]
define internal fastcc void @call5() {
  tail call fastcc void @call1()
  ret void
}

attributes #0 = {"amdgpu-flat-work-group-size"="1, 1024" "target-cpu"="gfx942" }
attributes #1 = {"amdgpu-flat-work-group-size"="1, 1024" "target-cpu"="gfx942" }
attributes #2 = {"amdgpu-flat-work-group-size"="1, 256" "target-cpu"="gfx942" }

; CHECK: attributes #[[ATTR0]] = { "amdgpu-agpr-alloc"="0" "amdgpu-flat-work-group-size"="1,256" "amdgpu-no-completion-action" "amdgpu-no-default-queue" "amdgpu-no-dispatch-id" "amdgpu-no-dispatch-ptr" "amdgpu-no-flat-scratch-init" "amdgpu-no-heap-ptr" "amdgpu-no-hostcall-ptr" "amdgpu-no-implicitarg-ptr" "amdgpu-no-lds-kernel-id" "amdgpu-no-multigrid-sync-arg" "amdgpu-no-queue-ptr" "amdgpu-no-workgroup-id-x" "amdgpu-no-workgroup-id-y" "amdgpu-no-workgroup-id-z" "amdgpu-no-workitem-id-x" "amdgpu-no-workitem-id-y" "amdgpu-no-workitem-id-z" "target-cpu"="gfx942" "uniform-work-group-size"="false" }
